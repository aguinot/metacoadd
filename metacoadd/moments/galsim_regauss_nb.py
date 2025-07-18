"""
This an implementation of the Galsim re-gauss algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp

This implementation is modified to allowed post-metacalibration PSF correction.
"""

from math import atan2, cos, exp, sin, sqrt, floor, ceil, log

import ngmix
import numpy as np
import pyfftw
from numba import njit, objmode

from .galsim_admom_nb import (
    find_ellipmom2,
    get_damped_intrinsic_moments,
    combine_multiband_observations_array,
    clear_result,
    clear_tmp,
    compute_effective_flux,
)


@njit(cache=True)
def _check_exp(pixels, w_data):
    w_sum = 0
    for pixel in pixels:
        val = ngmix.gmix.gmix_nb.gmix_eval_pixel_fast(w_data, pixel)
        w_sum += val
    return w_sum


@njit(cache=True)
def shearmult(e1_a, e2_a, e1_b, e2_b):
    dotp = e1_a * e1_b + e2_a * e2_b
    factor = (1.0 - sqrt(1 - e1_b * e1_b - e2_b * e2_b)) / (
        e1_b * e1_b + e2_b * e2_b
    )
    e1_out = (e1_a + e1_b + e2_b * factor * (e2_a * e1_b - e1_a * e2_b)) / (
        1 + dotp
    )
    e2_out = (e2_a + e2_b + e1_b * factor * (e1_a * e2_b - e2_a * e1_b)) / (
        1 + dotp
    )

    return e1_out, e2_out


@njit(cache=True)
def bj_nullPSF(T_ratio, e1_gal, e2_gal, rho4_gal, e1_psf, e2_psf, rho4_psf):
    cosheta_p = 1 / sqrt(1 - e1_psf * e1_psf - e2_psf * e2_psf)
    cosheta_g = 1 / sqrt(1 - e1_gal * e1_gal - e2_gal * e2_gal)
    sig2ratio = T_ratio * cosheta_g / cosheta_p

    e1_red, e2_red = shearmult(e1_gal, e2_gal, -e1_psf, -e2_psf)

    cosheta_g = 1 / sqrt(1 - e1_red * e1_red - e2_red * e2_red)
    R = 1.0 - sig2ratio * (1 + rho4_gal) / (1 - rho4_gal) / cosheta_g * (
        1 - rho4_psf
    ) / (1 + rho4_psf)

    e1_red /= R
    e2_red /= R

    e1_new, e2_new = shearmult(e1_red, e2_red, e1_psf, e2_psf)

    return e1_new, e2_new


@njit(cache=True)
def get_corrected_mom3(e1, e2, SB):
    """ """

    xx = 0.5 * sqrt(-((e1 + 1) ** 2) * SB**2 / (e1**2 + e2**2 - 1))
    yy = -xx * (e1 - 1) / (e1 + 1)
    xy = xx * e2 / (e1 + 1)

    return xx, yy, xy


@njit(cache=True)
def invert_BJ_correction(
    yy_final,
    xx_final,
    xy_final,
    T_psf,
    e1_psf,
    e2_psf,
    rho4gal,  # This is the M4/T^2 like quantity, e.g., 2 for a Gaussian
):
    """
    Inverts the sequence of operations to get xx_gal, yy_gal, xy_gal
    from xx_final, yy_final, xy_final.
    """

    # --- Step 1: Invert get_corrected_mom3 ---
    if (xx_final + yy_final) == 0:
        # This case needs careful handling, implies T_final = 0.
        return np.nan, np.nan, np.nan

    T_final = xx_final + yy_final
    e1_bj = (xx_final - yy_final) / T_final
    e2_bj = (2 * xy_final) / T_final

    e_bj_sq = e1_bj**2 + e2_bj**2
    if e_bj_sq >= 1.0:
        # Non-physical shear, SB_input would be NaN or complex
        return np.nan, np.nan, np.nan
    SB_input = T_final * sqrt(
        1.0 - e_bj_sq
    )  # This is T_gal_obs / cosheta_g_obs = T_gal_intrinsic_projected

    if SB_input < 1e-9:  # Changed from == 0 to < 1e-9 for robustness
        # This implies T_gal_intrinsic_projected is zero.
        # If T_gal is 0, then xx_gal = yy_gal = xy_gal = 0.
        # This path assumes if SB_input is effectively zero, T_gal is zero.
        if T_final < 1e-9:  # If input moments (and thus T_final) are zero
            return 0.0, 0.0, 0.0
        # If T_final is not zero but SB_input is, implies e_bj_sq is 1.0 (already checked)
        # or T_final is small and numerical precision issues.
        return np.nan, np.nan, np.nan

    # --- Step 2: Invert bj_nullPSF ---
    # Constants for PSF
    e_psf_sq = e1_psf**2 + e2_psf**2
    if e_psf_sq >= 1.0:  # Unphysical PSF
        return np.nan, np.nan, np.nan
    cosheta_p = 1.0 / sqrt(1.0 - e_psf_sq)

    # Calculate C1 (which is C_coeff in derivation: sig2ratio * k_eff)
    # sig2ratio = (T_psf / cosheta_p) / SB_input
    # k_eff = rho4gal / (4.0 - rho4gal)

    sig2ratio_val = 0.0
    if cosheta_p == 0:  # Should not happen due to e_psf_sq check
        return np.nan, np.nan, np.nan
    # SB_input non-zero check is implicitly handled by earlier SB_input < 1e-9 check
    sig2ratio_val = T_psf / (SB_input * cosheta_p)

    k_eff_numerator = rho4gal
    k_eff_denominator = 4.0 - rho4gal

    C1 = 0.0
    if np.isfinite(k_eff_numerator) and np.isfinite(k_eff_denominator):
        if abs(k_eff_denominator) < 1e-9:  # rho4gal is approx 4.0
            # If rho4gal = 4, K_eff is infinite.
            # R = 1 - (sig2ratio_val / cosheta_g_red) * K_eff
            # If sig2ratio_val is non-zero, R -> -inf (unphysical).
            # If sig2ratio_val is zero (e.g. T_psf=0), then R=1, C1=0.
            if (
                abs(sig2ratio_val * k_eff_numerator) < 1e-9
            ):  # Effectively T_psf=0 or rho4gal=0 (if rho4gal=4, means T_psf=0)
                C1 = 0.0  # Results in R_factor = 1.0
            else:  # rho4gal=4 and T_psf !=0: K_eff is infinite, C1 is infinite. Problematic.
                # Instead of returning nan immediately, we can use a capped C1.
                # Or, if truly infinite k_eff is intended to lead to failure, keep nan.
                # For stability, let's assume extreme C1 is the problem.
                # If sig2ratio_val * k_eff_numerator is positive, C1 -> +inf
                # If sig2ratio_val * k_eff_numerator is negative, C1 -> -inf
                # Capping C1 will be done after this block.
                # For now, if it's this specific singularity, returning nan might be intended.
                return np.nan, np.nan, np.nan
        elif abs(k_eff_numerator) < 1e-9:  # rho4gal is approx 0.0
            C1 = 0.0  # K_eff is 0, so C1 is 0. Results in R_factor = 1.0
        else:
            k_eff = k_eff_numerator / k_eff_denominator
            C1 = sig2ratio_val * k_eff

    # Cap the magnitude of C1 to prevent numerical instability
    C1_MAX_MAG = 500.0  # This value may need tuning
    if C1 > C1_MAX_MAG:
        C1 = C1_MAX_MAG
    elif C1 < -C1_MAX_MAG:
        C1 = -C1_MAX_MAG

    # Sub-step a (from bj_nullPSF, but e1_red, e2_red are scaled by R in bj_nullPSF)
    # Here, e1_red_scaled, e2_red_scaled are e_red / R from bj_nullPSF perspective
    # Or, e1_red_scaled, e2_red_scaled are the result of shearmult(e_new, e_new_conj, e_psf_conj)
    e1_red_div_R, e2_red_div_R = shearmult(
        e1_bj, e2_bj, -e1_psf, -e2_psf
    )  # These are e1_red/R, e2_red/R
    e_red_div_R_sq = e1_red_div_R**2 + e2_red_div_R**2

    if (
        e_red_div_R_sq >= 1.0 + 1e-9
    ):  # e_red/R cannot be > 1 if R is physical and e_red < 1
        # Allow for small numerical errors if e_red_div_R_sq is slightly above 1
        # if R is very small.
        # This check might be too strict if R is very small.
        # However, if e_red_div_R_sq > 1, then 1 - R^2 * e_red_div_R_sq can be negative.
        pass  # This will be caught by discriminant_quad < 0 if it leads to issues.

    # Sub-step d: Solve for R_factor (R in bj_nullPSF)
    # R_factor^2 * A_quad + R_factor * B_quad + C_quad_val = 0
    # A_quad = 1.0 + C1**2 * e_red_div_R_sq
    # B_quad = -2.0
    # C_quad_val = 1.0 - C1**2

    A_quad = 1.0 + C1**2 * e_red_div_R_sq
    B_quad = -2.0
    C_quad_val = 1.0 - C1**2

    discriminant_quad = B_quad**2 - 4.0 * A_quad * C_quad_val
    if discriminant_quad < -1e-9:  # Allow for small negative due to precision
        return np.nan, np.nan, np.nan  # No real solution for R_factor
    if discriminant_quad < 0:
        discriminant_quad = 0  # Clamp small negatives

    R_factor = 0.0
    # The condition for choosing roots: (R-1) + C1 * sqrt(1 - R^2 * (e_red/R)^2) = 0
    # sqrt(1 - R^2 * e_red_div_R_sq) is cosh_eta_g_red_inv = 1 / cosh(eta_g_red)
    # So (R-1) + C1 / cosh(eta_g_red(R)) = 0

    # When C1 = 0, R_factor must be 1.
    # A_quad = 1, B_quad = -2, C_quad_val = 1. disc = 4 - 4*1*1 = 0. R = 2/2 = 1.
    if abs(C1) < 1e-9:  # Effectively C1 is zero
        R_factor = 1.0
    else:
        sqrt_disc = sqrt(discriminant_quad)
        # Standard quadratic formula roots for A*R^2 + B*R + C = 0 are (-B +/- sqrt(disc))/(2A)
        r1 = (-B_quad + sqrt_disc) / (2.0 * A_quad)
        r2 = (-B_quad - sqrt_disc) / (2.0 * A_quad)

        chosen_R = False
        # Try r1
        # val_r1_term is (1 - r1^2 * e_red_div_R_sq) which is (1/cosh(eta_g_red(r1)))^2
        val_r1_term = 1.0 - r1**2 * e_red_div_R_sq
        if val_r1_term >= -1e-9:  # check if 1/cosh^2 is valid (non-negative)
            # Check if (r1-1) + C1 * sqrt(max(0, val_r1_term)) approx 0
            if abs((r1 - 1.0) + C1 * sqrt(max(0, val_r1_term))) < 1e-7:
                R_factor = r1
                chosen_R = True

        if not chosen_R:
            # Try r2
            val_r2_term = 1.0 - r2**2 * e_red_div_R_sq
            if val_r2_term >= -1e-9:
                if abs((r2 - 1.0) + C1 * sqrt(max(0, val_r2_term))) < 1e-7:
                    R_factor = r2
                    chosen_R = True

        if not chosen_R:
            # Fallback if neither root strictly satisfies the check,
            # could be due to numerical precision.
            # The original GalSim HSM code has a preferred root choice based on sign of C1.
            # (R-1) should have opposite sign to C1 (or be zero).
            # If C1 > 0, R < 1. If C1 < 0, R > 1.
            # r_alt1 = (1.0 + C1 * sqrt(max(0,1.0 - e_red_div_R_sq + C1**2 * e_red_div_R_sq))) / A_quad
            # r_alt2 = (1.0 - C1 * sqrt(max(0,1.0 - e_red_div_R_sq + C1**2 * e_red_div_R_sq))) / A_quad
            # The term sqrt(1 - e_red_div_R_sq + C1^2 * e_red_div_R_sq) is sqrt_disc / (2*abs(C1)) if C1!=0
            # So r_alt1,2 = (1 +/- 0.5 * sqrt_disc) / A_quad. This matches -B_quad / (2*A_quad) +/- sqrt_disc / (2*A_quad)
            # which are r1 and r2.
            # The choice often depends on which solution is closer to 1 or physical.
            # If C1 > 0, we expect R < 1. r2 is typically smaller.
            # If C1 < 0, we expect R > 1. r1 is typically larger.
            # Let's check physical R > 0.
            if r1 > 0 and r2 <= 0:
                R_factor = r1
                chosen_R = True
            elif r2 > 0 and r1 <= 0:
                R_factor = r2
                chosen_R = True
            elif r1 > 0 and r2 > 0:  # Both positive, pick based on C1
                if C1 > 0:
                    R_factor = min(r1, r2)  # Expect R < 1
                else:
                    R_factor = max(r1, r2)  # Expect R > 1
                # Check if this choice is consistent with the val_term
                val_rf_term = 1.0 - R_factor**2 * e_red_div_R_sq
                if (
                    val_rf_term < -1e-9
                ):  # This choice makes cosh_eta_g_red imaginary
                    return np.nan, np.nan, np.nan
                chosen_R = True

        if not chosen_R:
            return np.nan, np.nan, np.nan  # No consistent root found

    if R_factor <= 1e-9:  # R must be positive, and typically around 1.
        return np.nan, np.nan, np.nan

    # Sub-step e (invert e_red_scaled = e_red / R)
    e1_red = R_factor * e1_red_div_R
    e2_red = R_factor * e2_red_div_R

    # Sub-step f (invert e1_red, e2_red = shearmult(e1_gal, e2_gal, -e1_psf, -e2_psf))
    # So, e1_gal, e2_gal = shearmult(e1_red, e2_red, e1_psf, e2_psf) (since -(-e1_psf) = e1_psf)
    e1_gal, e2_gal = shearmult(e1_red, e2_red, e1_psf, e2_psf)

    # Sub-step g (calculate T_gal from SB_input and e1_gal, e2_gal)
    e_gal_sq = e1_gal**2 + e2_gal**2
    if e_gal_sq >= 1.0 - 1e-9:  # Allow for small numerical error
        # This implies intrinsic galaxy is maximally sheared or unphysical.
        # If SB_input was non-zero, this is problematic.
        # If SB_input was (close to) zero, T_gal should be zero.
        if SB_input < 1e-9:
            T_gal = 0.0
            # If T_gal is 0, then moments are 0
            return 0.0, 0.0, 0.0
        else:  # SB_input is non-zero but e_gal_sq is ~1, inconsistent for finite T_gal
            return np.nan, np.nan, np.nan

    cosheta_g_obs_equiv = 1.0 / sqrt(
        max(1e-9, 1.0 - e_gal_sq)
    )  # cosheta for the original galaxy
    T_gal = (
        SB_input * cosheta_g_obs_equiv
    )  # SB_input = T_gal_obs / cosheta_g_obs

    # --- Step 3: Recover xx_gal, yy_gal, xy_gal ---
    # Note: T_gal here is T_gal_obs (sum of second moments, not intrinsic size)
    xx_gal = 0.5 * T_gal * (1.0 + e1_gal)
    yy_gal = 0.5 * T_gal * (1.0 - e1_gal)
    xy_gal = 0.5 * T_gal * e2_gal

    return xx_gal, xy_gal, yy_gal


@njit(cache=True)
def BJ_correction(yy_gal, xy_gal, xx_gal, rho4gal, e1_psf, e2_psf, T_psf):
    T_gal = xx_gal + yy_gal
    e1_gal = (xx_gal - yy_gal) / T_gal
    e2_gal = (2 * xy_gal) / T_gal

    e1_bj, e2_bj = bj_nullPSF(
        T_psf / T_gal, e1_gal, e2_gal, 0.5 * rho4gal - 1, e1_psf, e2_psf, 0
    )

    size_scaling = sqrt(1 - (e1_gal**2 + e2_gal**2))
    xx_final, yy_final, xy_final = get_corrected_mom3(
        e1_bj, e2_bj, T_gal * size_scaling
    )

    return xx_final, xy_final, yy_final


@njit(cache=False)
def find_ellipmom1_bj(
    pixels_list,
    band_tracker,
    x0,
    y0,
    Mxx_,
    Mxy_,
    Myy_,
    res,
    tmp,
    conf,
    do_cov=False,
    psf_moments=None,
    rho4gal=None,
):
    F = res["F"]

    n_bands = len(band_tracker)
    if not do_cov:
        flux_weights = np.zeros(n_bands, dtype=np.float64)

    tmp_sums = np.zeros(6 + n_bands, dtype=np.float64)

    tracking = 0
    band_ind = 0
    n_list = len(pixels_list)
    for i_list in range(n_list):
        T_psf = psf_moments[i_list][0] + psf_moments[i_list][2]
        e1_psf = (psf_moments[i_list][2] - psf_moments[i_list][0]) / T_psf
        e2_psf = 2 * psf_moments[i_list][1] / T_psf

        T_gal = Mxx_ + Myy_
        T_obs = T_gal + T_psf
        R = 1 - T_psf / T_obs / sqrt(1 - e1_psf**2 - e2_psf**2)
        e1_gal = (Myy_ - Mxx_) / T_obs / R
        e2_gal = (2 * Mxy_) / T_obs / R

        Mxx__ = 0.5 * T_obs * (1 - e1_gal)
        Myy__ = 0.5 * T_obs * (1 + e1_gal)
        Mxy__ = 0.5 * T_obs * e2_gal

        Myy, Mxy, Mxx = invert_BJ_correction(
            Mxx__,
            Myy__,
            Mxy__,
            T_psf,
            e1_psf,
            e2_psf,
            rho4gal,
        )

        if (
            (not np.isfinite(Mxx))
            | (not np.isfinite(Myy))
            | (not np.isfinite(Mxy))
        ):
            tracking += 1
            if tracking == band_tracker[band_ind]:
                band_ind += 1
                tracking = 0
            continue

        # We need to compute the normalization before without the PSF correction
        detM = Mxx * Myy - Mxy * Mxy
        w_norm = 1.0 / (2 * np.pi * sqrt(detM))
        res["wnorm"] = 1.0

        Minv_xx = Myy / detM
        TwoMinv_xy = -Mxy / detM * 2.0
        Minv_yy = Mxx / detM

        # For computing the flux Jacobian wrt Q = (Q11, Q22, Q12)
        Minv_00 = Minv_xx
        Minv_01 = -0.5 * TwoMinv_xy
        Minv_10 = Minv_01
        Minv_11 = Minv_yy

        pixels = pixels_list[i_list]
        tmp_sums[:] = 0.0
        ivar_sum = 0.0
        n_pixels = pixels.size
        for i_pix in range(n_pixels):
            pixel = pixels[i_pix]

            ivar = pixel["ierr"] * pixel["ierr"]

            umod = pixel["u"] - x0
            vmod = pixel["v"] - y0

            Minv_xx__x_x0__x_x0 = Minv_xx * vmod * vmod
            TwoMinv_xy__y_y0__x_x0 = TwoMinv_xy * vmod * umod
            Minv_yy__y_y0__y_y0 = Minv_yy * umod * umod

            rho2 = (
                Minv_yy__y_y0__y_y0
                + TwoMinv_xy__y_y0__x_x0
                + Minv_xx__x_x0__x_x0
            )

            res["npix"] += 1
            if rho2 < conf["max_moment_nsig2"]:
                win = exp(-0.5 * rho2) * w_norm * pixel["area"]
                intensity = win * pixel["val"]

                res["wsum"] += win
                ivar_sum += ivar

                if not do_cov:
                    tmp_sums[0] += umod * intensity
                    tmp_sums[1] += vmod * intensity
                    tmp_sums[2] += vmod * vmod * intensity
                    tmp_sums[3] += umod * vmod * intensity
                    tmp_sums[4] += umod * umod * intensity
                    tmp_sums[5] += rho2 * rho2 * intensity
                    tmp_sums[6 + band_ind] += 1.0 * intensity
                else:
                    # Accumulate flux Jacobian w.r.t Q = (Q11, Q22, Q12)
                    # These are symmetric matrix derivatives
                    Minv_r0 = Minv_00 * umod + Minv_01 * vmod
                    Minv_r1 = Minv_10 * umod + Minv_11 * vmod

                    # rᵀ Minv dM Minv r
                    d_rho2_dQ11 = -(Minv_r1 * Minv_r1)
                    d_rho2_dQ22 = -(Minv_r0 * Minv_r0)
                    d_rho2_dQ12 = -(2.0 * Minv_r0 * Minv_r1)

                    # Tr(Minv @ dM): for symmetric matrices
                    tr_Q11 = Minv_11
                    tr_Q22 = Minv_00
                    tr_Q12 = 2.0 * Minv_01

                    # d log(w) / dQk = 0.5 * (rᵀ Minv dM Minv r - Tr(Minv dM))
                    dlogw_dQ11 = 0.5 * (-(d_rho2_dQ11) - tr_Q11)
                    dlogw_dQ22 = 0.5 * (-(d_rho2_dQ22) - tr_Q22)
                    dlogw_dQ12 = 0.5 * (-(d_rho2_dQ12) - tr_Q12)

                    tmp["flux_jac"][i_list][0] -= intensity * Minv_r0  # dF/dx0  # fmt: skip
                    tmp["flux_jac"][i_list][1] -= intensity * Minv_r1  # dF/dy0  # fmt: skip
                    tmp["flux_jac"][i_list][2] += (intensity * dlogw_dQ22)  # dF/dQ11  # fmt: skip
                    tmp["flux_jac"][i_list][3] += (intensity * dlogw_dQ12)  # dF/dQ12  # fmt: skip
                    tmp["flux_jac"][i_list][4] += (intensity * dlogw_dQ11)  # dF/dQ22  # fmt: skip
                    # tmp["flux_jac"][i_list][5] += 0                        # dF/drho2  # fmt: skip

                    win2 = win * win
                    var = 1.0 / ivar
                    F[0] = umod
                    F[1] = vmod
                    F[2] = vmod * vmod
                    F[3] = umod * vmod
                    F[4] = umod * umod
                    F[5] = rho2 * rho2
                    F[6] = 1.0
                    for i in range(7):
                        tmp["sums"][i_list][i] += intensity * F[i]
                        for j in range(7):
                            tmp["sums_cov"][i_list][i, j] += (
                                win2 * var * F[i] * F[j]
                            )

        if not do_cov:
            tmp_sums[:6] /= tmp_sums[6 + band_ind]
            new_yy, new_xy, new_xx = BJ_correction(
                tmp_sums[2] * 2,
                tmp_sums[3] * 2,
                tmp_sums[4] * 2,
                tmp_sums[5],
                e1_psf,
                e2_psf,
                T_psf,
            )
            T_BJ = new_xx + new_yy

            T_gal = T_BJ - T_psf
            R = 1 - T_psf / T_BJ / sqrt(1 - e1_psf**2 - e2_psf**2)
            e1_gal = (new_yy - new_xx) / T_gal * R
            e2_gal = (2 * new_xy) / T_gal * R

            new_xx = 0.5 * T_gal * (1 - e1_gal)
            new_yy = 0.5 * T_gal * (1 + e1_gal)
            new_xy = 0.5 * T_gal * e2_gal
            e1_gal = (new_yy - new_xx) / T_gal
            e2_gal = (2 * new_xy) / T_gal

            tmp_sums[2:5] = new_xx, new_xy, new_yy
            tmp_sums[2:5] /= 2

            res["sums"][:] += tmp_sums * ivar_sum
            flux_weights[band_ind] += ivar_sum
        else:
            tmp["sums"][i_list][:6] /= tmp["sums"][i_list][6]
            new_yy, new_xy, new_xx = BJ_correction(
                tmp["sums"][i_list][2] * 2,
                tmp["sums"][i_list][3] * 2,
                tmp["sums"][i_list][4] * 2,
                tmp["sums"][i_list][5],
                e1_psf,
                e2_psf,
                T_psf,
            )
            T_BJ = new_xx + new_yy

            T_gal = T_BJ - T_psf
            R = 1 - T_psf / T_BJ / sqrt(1 - e1_psf**2 - e2_psf**2)
            e1_gal = (new_yy - new_xx) / T_gal * R
            e2_gal = (2 * new_xy) / T_gal * R

            new_xx = 0.5 * T_gal * (1 - e1_gal)
            new_yy = 0.5 * T_gal * (1 + e1_gal)
            new_xy = 0.5 * T_gal * e2_gal

            tmp["sums"][i_list][2:5] = new_xx, new_xy, new_yy
            tmp["sums"][i_list][2:5] /= 2
            tmp["sums"][i_list][6] /= w_norm
        tracking += 1
        if tracking == band_tracker[band_ind]:
            band_ind += 1
            tracking = 0

    if do_cov:
        combine_multiband_observations_array(res, tmp, band_tracker)

        snr = np.sqrt(
            res["sums"] @ np.linalg.inv(res["sums_cov"]) @ res["sums"]
        )
        res["s2n"] = snr
    else:
        if np.sum(flux_weights) != 0:
            res["sums"][:] /= np.sum(flux_weights)
        else:
            res["sums"][:] = np.nan

    res["wsum"] /= n_list


@njit(cache=False)
def find_ellipmom2_bj(
    pixels_list,
    band_tracker,
    guess,
    resarray,
    tmparray,
    confarray,
    psf_moments=None,
):
    """ """

    conf = confarray[0]
    res = resarray[0]
    tmp = tmparray[0]

    convergence_factor = 1.0
    shiftscale0 = 0.0

    pix_scale = np.sqrt(np.mean(pixels_list[0]["area"]))

    x0, y0, Mxx, Mxy, Myy = guess
    rho4 = 2.0

    x00 = x0
    y00 = y0
    do_cov = False

    # Prepare the effective PSF moments for deconvolution
    psf_moms_for_ellipmom1 = None
    fixed_psf_components = np.array([0.0, 0.0, 0.0])
    if psf_moments is not None:  # true_psf_moments are provided
        fixed_psf_components = np.array([0.0121, 0.0, 0.0121])
        # fixed_psf_components = np.array([0.0, 0.0, 0.0])
        if fixed_psf_components is not None:
            # Calculate M_psf_effective = M_psf_true - M_psf_fixed for each observation
            psf_moms_for_ellipmom1 = psf_moments - fixed_psf_components
            # for true_psf_obs_i in psf_moments:
            #     # Ensure fixed_psf_components is a NumPy array for element-wise subtraction
            #     psf_moms_for_ellipmom1.append(
            #         true_psf_obs_i
            #         - np.array(fixed_psf_components, dtype=np.float64)
            #     )
        else:
            # Original behavior: deconvolve by the true PSF
            psf_moms_for_ellipmom1 = psf_moments
    # If psf_moments is None, psf_moms_for_ellipmom1 remains None (no PSF deconvolution)

    for i in range(conf["maxiter"]):
        clear_result(res)
        clear_tmp(tmp)
        find_ellipmom1_bj(
            pixels_list,
            band_tracker,
            x0,
            y0,
            Mxx,
            Mxy,
            Myy,
            res,
            tmp,
            conf,
            do_cov,
            psf_moments=psf_moms_for_ellipmom1,
            rho4gal=rho4,
        )
        Bx, By, Cxx, Cxy, Cyy, rho4 = res["sums"][:6]
        Amps = res["sums"][6:]
        if do_cov:
            Amp, _, _, _ = compute_effective_flux(
                Amps, res["sums_cov"][6:, 6:]
            )
        else:
            Amp = sum(Amps)

        if Amp <= 0:
            res["flags"] = ngmix.flags.NONPOS_FLUX

        two_psi = atan2(2 * Mxy, Mxx - Myy)
        semi_a2 = 0.5 * ((Mxx + Myy) + (Mxx - Myy) * cos(two_psi)) + (
            Mxy * sin(two_psi)
        )
        semi_b2 = Mxx + Myy - semi_a2

        if semi_b2 <= 0:
            res["flags"] = ngmix.flags.NONPOS_SIZE

        if semi_b2 <= 0:
            shiftscale = np.nan
        else:
            shiftscale = sqrt(semi_b2)
        if res["numiter"] == 0:
            shiftscale0 = shiftscale

        dx = 2.0 * Bx / (1.0 * shiftscale)
        dy = 2.0 * By / (1.0 * shiftscale)
        dxx = 4 * (Cxx / 1.0 - 0.5 * Mxx) / semi_b2
        dxy = 4 * (Cxy / 1.0 - 0.5 * Mxy) / semi_b2
        dyy = 4 * (Cyy / 1.0 - 0.5 * Myy) / semi_b2

        if dx > conf["bound_correct_wt"]:
            dx = conf["bound_correct_wt"]
        if dx < -conf["bound_correct_wt"]:
            dx = -conf["bound_correct_wt"]
        if not np.isfinite(dx):
            dx = conf["bound_correct_wt"]
        if dy > conf["bound_correct_wt"]:
            dy = conf["bound_correct_wt"]
        if dy < -conf["bound_correct_wt"]:
            dy = -conf["bound_correct_wt"]
        if not np.isfinite(dy):
            dy = conf["bound_correct_wt"]
        if dxx > conf["bound_correct_wt"]:
            dxx = conf["bound_correct_wt"]
        if dxx < -conf["bound_correct_wt"]:
            dxx = -conf["bound_correct_wt"]
        if not np.isfinite(dxx):
            dxx = conf["bound_correct_wt"]
        if dxy > conf["bound_correct_wt"]:
            dxy = conf["bound_correct_wt"]
        if dxy < -conf["bound_correct_wt"]:
            dxy = -conf["bound_correct_wt"]
        if not np.isfinite(dxy):
            dxy = conf["bound_correct_wt"]
        if dyy > conf["bound_correct_wt"]:
            dyy = conf["bound_correct_wt"]
        if dyy < -conf["bound_correct_wt"]:
            dyy = -conf["bound_correct_wt"]
        if not np.isfinite(dyy):
            dyy = conf["bound_correct_wt"]

        if abs(dx) > abs(dy):
            convergence_factor = abs(dx)
        else:
            convergence_factor = abs(dy)
        convergence_factor = convergence_factor * convergence_factor

        if abs(dxx) > convergence_factor:
            convergence_factor = abs(dxx)
        if abs(dxy) > convergence_factor:
            convergence_factor = abs(dxy)
        if abs(dyy) > convergence_factor:
            convergence_factor = abs(dyy)

        convergence_factor = sqrt(convergence_factor)
        if shiftscale < shiftscale0:
            convergence_factor *= shiftscale0 / shiftscale

        x0 += dx * shiftscale
        y0 += dy * shiftscale
        Mxx += dxx * semi_b2
        Mxy += dxy * semi_b2
        Myy += dyy * semi_b2

        if (abs(x0 - x00) > conf["shiftmax"] * pix_scale) | (
            abs(y0 - y00) > conf["shiftmax"] * pix_scale
        ):
            res["flags"] = ngmix.flags.CEN_SHIFT

        res["numiter"] = i + 1

        if convergence_factor < conf["tol"]:  # or do_cov:
            if not do_cov:
                do_cov = True
                continue
            Mxx_corr, Mxy_corr, Myy_corr = get_damped_intrinsic_moments(
                Mxx,
                Mxy,
                Myy,
                fixed_psf_components,
                conf["lambda_ell"],
                conf["min_T_abs"],
            )
            # rho4 /= Amp
            res["pars"][0] = x0
            res["pars"][1] = y0
            # res["pars"][2] = Mxx - fixed_psf_components[0]
            # res["pars"][3] = Mxy - fixed_psf_components[1]
            # res["pars"][4] = Myy - fixed_psf_components[2]
            res["pars"][2] = Mxx_corr
            res["pars"][3] = Mxy_corr
            res["pars"][4] = Myy_corr
            res["pars"][5] = rho4
            res["pars"][6:] = Amps * rho4
            break

        if res["numiter"] == conf["maxiter"]:
            res["flags"] = ngmix.flags.MAXITER


@njit(cache=True)
def goodFFTSize(N):
    if N <= 2:
        return 2
    # Reduce slightly to eliminate potential rounding errors:
    insize = (1.0 - 1e-5) * N
    log2n = log(2.0) * ceil(log(insize) / log(2.0))
    log2n3 = log(3.0) + log(2.0) * ceil((log(insize) - log(3.0)) / log(2.0))
    # must be even number
    log2n3 = max(log2n3, log(6.0))
    Nk = int(ceil(exp(min(log2n, log2n3)) - 1e-5))
    return Nk


@njit(cache=True)
def fast_convolve_image1(
    image1, image2, image_out, orig_img1=(0, 0), orig_img2=(0, 0)
):
    # Input
    N1 = int(max(image1.shape) * 4 / 3)
    N2 = int(max(image2.shape) * 4 / 3)
    N3 = int(max(image_out.shape))
    N = int(max((N1, N2, N3)))
    N = goodFFTSize(N)

    # Make NxN image
    xim = np.zeros((N, N))
    offset = int(N / 4)
    b1 = [
        offset,
        image1.shape[0] + offset - 1,
        offset,
        image1.shape[1] + offset - 1,
    ]
    xim[b1[0] : b1[1] + 1, b1[2] : b1[3] + 1] = image1

    # Do fft img1
    kb = (int(N / 2), N - 1)
    with objmode(kim1="complex128[:,:]"):
        xim_pyfft = pyfftw.empty_aligned(xim.shape, dtype="float64")
        xim_pyfft[:] = xim
        fft_object_xim1 = pyfftw.builders.rfft2(xim_pyfft, s=(N, N))
        kim1 = fft_object_xim1()

    # Do fft img2
    xim.fill(0)
    b2 = [
        offset,
        image2.shape[0] + offset - 1,
        offset,
        image2.shape[1] + offset - 1,
    ]
    xim[b2[0] : b2[1] + 1, b2[2] : b2[3] + 1] = image2
    with objmode(kim2="complex128[:,:]"):
        xim_pyfft = pyfftw.empty_aligned(xim.shape, dtype="float64")
        xim_pyfft[:] = xim
        fft_object_xim2 = pyfftw.builders.rfft2(xim_pyfft, s=(N, N))
        kim2 = fft_object_xim2()

    # Conv
    kim2 *= kim1

    # Inverse fft
    kb = (N, int(N / 2) + 1)
    with objmode(xim="float64[:,:]"):
        kim_pyfft = pyfftw.empty_aligned(kb, dtype="complex128")
        kim_pyfft[:] = kim2
        ifft_object_out = pyfftw.builders.irfft2(kim_pyfft, s=(N, N))
        xim = ifft_object_out()
        xim = np.fft.fftshift(xim).real

    shift_x = orig_img1[0] + orig_img2[0]
    shift_y = orig_img1[1] + orig_img2[1]
    b3 = [
        -shift_x,
        image_out.shape[0] - shift_x,
        -shift_y,
        image_out.shape[1] - shift_y,
    ]
    if b3[0] < 0:
        b3[0] = 0
    if b3[1] > xim.shape[0]:
        b3[1] = xim.shape[0] - 1
    if b3[2] < 0:
        b3[2] = 0
    if b3[3] > xim.shape[1]:
        b3[3] = xim.shape[1] - 1
    b4 = [b3[0] + shift_x, b3[1] + shift_x, b3[2] + shift_y, b3[3] + shift_y]
    image_out[b4[0] : b4[1], b4[2] : b4[3]] += xim[
        b3[0] : b3[1], b3[2] : b3[3]
    ]


def get_resi_img(
    obs,
    xx_f,
    xy_f,
    yy_f,
    flux_gal,
    xx_psf,
    xy_psf,
    yy_psf,
    flux_psf,
):
    nsig_rg = 3.0
    nsig_rg2 = 3.6

    x_gal_min, y_gal_min = 0, 0
    x_gal_max, y_gal_max = obs.image.shape
    x_psf_min, y_psf_min = 0, 0
    x_psf_max, y_psf_max = obs.psf.image.shape

    if xx_f <= obs.jacobian.area:
        xx_f = obs.jacobian.area
    if yy_f <= obs.jacobian.area:
        yy_f = obs.jacobian.area

    # Get fgauss bounds
    fgauss_xmin = x_gal_min - x_psf_max
    fgauss_xmax = x_gal_max - x_psf_min
    fgauss_ymin = y_gal_min - y_psf_max
    fgauss_ymax = y_gal_max - y_psf_min
    fgauss_xctr = obs.jacobian.row0 - obs.psf.jacobian.row0
    fgauss_yctr = obs.jacobian.col0 - obs.psf.jacobian.col0
    fgauss_xsig = np.sqrt(xx_f / obs.jacobian.area)
    fgauss_ysig = np.sqrt(yy_f / obs.jacobian.area)
    if fgauss_xmin < fgauss_xctr - nsig_rg * fgauss_xsig:
        fgauss_xmin = int(floor(fgauss_xctr - nsig_rg * fgauss_xsig))
    if fgauss_xmax > fgauss_xctr + nsig_rg * fgauss_xsig:
        fgauss_xmax = int(ceil(fgauss_xctr + nsig_rg * fgauss_xsig))
    if fgauss_ymin < fgauss_yctr - nsig_rg * fgauss_ysig:
        fgauss_ymin = int(floor(fgauss_yctr - nsig_rg * fgauss_ysig))
    if fgauss_ymax > fgauss_yctr + nsig_rg * fgauss_ysig:
        fgauss_ymax = int(ceil(fgauss_yctr + nsig_rg * fgauss_ysig))
    f_dim_x = fgauss_xmax - fgauss_xmin
    f_dim_y = fgauss_ymax - fgauss_ymin
    f_row0 = f_dim_x / 2
    f_col0 = f_dim_y / 2
    f_dim_x += 1
    f_dim_y += 1
    f_jac = ngmix.Jacobian(
        row=f_row0, col=f_col0, wcs=obs.jacobian.get_galsim_wcs()
    )

    # Get PSF bounds
    p_xmin = int(
        floor(
            obs.psf.jacobian.row0
            - nsig_rg2 * sqrt(xx_f / obs.jacobian.area)
            - nsig_rg * fgauss_xsig
        )
    )
    p_xmax = int(
        ceil(
            obs.psf.jacobian.row0
            + nsig_rg2 * sqrt(xx_f / obs.jacobian.area)
            + nsig_rg * fgauss_xsig
        )
    )
    p_ymin = int(
        floor(
            obs.psf.jacobian.col0
            - nsig_rg2 * sqrt(yy_f / obs.jacobian.area)
            - nsig_rg * fgauss_ysig
        )
    )
    p_ymax = int(
        ceil(
            obs.psf.jacobian.col0
            + nsig_rg2 * sqrt(yy_f / obs.jacobian.area)
            + nsig_rg * fgauss_ysig
        )
    )
    if x_psf_min >= p_xmin:
        p_xmin = x_psf_min
    if x_psf_max <= p_xmax:
        p_xmax = x_psf_max
    if y_psf_min >= p_ymin:
        p_ymin = y_psf_min
    if y_psf_max <= p_ymax:
        p_ymax = y_psf_max
    p_dim_x = p_xmax - p_xmin
    p_dim_y = p_ymax - p_ymin
    p_row0 = p_dim_x / 2
    p_col0 = p_dim_y / 2
    p_dim_x += 1
    p_dim_y += 1
    p_jac = ngmix.Jacobian(
        row=p_row0, col=p_col0, wcs=obs.psf.jacobian.get_galsim_wcs()
    )

    g1, g2, T = ngmix.moments.mom2g(yy_f, xy_f, xx_f)
    pars_fgauss = np.zeros(6)
    pars_fgauss[2] = g1
    pars_fgauss[3] = g2
    pars_fgauss[4] = T
    pars_fgauss[5] = flux_gal
    gmix_fgauss = ngmix.GMixModel(pars_fgauss, "gauss")
    fgauss_img = gmix_fgauss.make_image(
        (f_dim_x, f_dim_y), f_jac, fast_exp=True
    )

    g1_psf, g2_psf, T_psf = ngmix.moments.mom2g(yy_psf, xy_psf, xx_psf)
    pars_psf = np.zeros(6)
    pars_psf[2] = g1_psf
    pars_psf[3] = g2_psf
    pars_psf[4] = T_psf
    pars_psf[5] = flux_psf
    gmix_psf = ngmix.GMixModel(pars_psf, "gauss")
    fpsf_img = gmix_psf.make_image((p_dim_x, p_dim_y), p_jac, fast_exp=True)

    PSF_resid_img = (
        -obs.psf.image[p_xmin : p_xmax + 1, p_ymin : p_ymax + 1] + fpsf_img
    )

    fgauss_img *= flux_gal / np.sum(fgauss_img)

    out_image_img = obs.image.copy()

    fast_convolve_image1(
        fgauss_img,
        PSF_resid_img,
        out_image_img,
        orig_img1=(fgauss_xmin, fgauss_ymin),
        orig_img2=(p_xmin, p_ymin),
    )

    return out_image_img


def get_true_resi_img(
    obs,
    x0_gal,
    y0_gal,
    xx_f,
    yy_f,
    xy_f,
    flux_gal,
    psf_resi=None,
):
    # Approx deconv
    if xx_f <= obs.jacobian.area:
        xx_f = obs.jacobian.area
    if yy_f <= obs.jacobian.area:
        yy_f = obs.jacobian.area

    g1, g2, T = ngmix.moments.mom2g(yy_f, xy_f, xx_f)

    if psf_resi is not None:
        import galsim

        f = (
            galsim.Gaussian(sigma=np.sqrt(T / 2))
            .shear(g1=g1, g2=g2)
            .withFlux(flux_gal)
        )
        wcs_loc = obs.jacobian.get_galsim_wcs()
        f_conv = galsim.Convolve(
            f,
            psf_resi,
        )
        row_shift, col_shift = obs.jacobian.get_rowcol(y0_gal, x0_gal)
        nrow, ncol = obs.image.shape
        canonical_center = (np.array((ncol, nrow)) - 1.0) / 2.0
        offset = (col_shift, row_shift) - canonical_center
        f_conv_img = f_conv.drawImage(
            nx=ncol,
            ny=nrow,
            wcs=wcs_loc,
            method="no_pixel",
            offset=offset,
        )
        out_image_img2 = obs.image.copy()
        out_image_img2 += f_conv_img.array

    return out_image_img2


def regauss(
    mbobs,
    guess,
    resarray,
    tmp_func,
    confarray,
):
    pixels_list = []
    band_tracker = []
    psf_moments = []
    idx = 0
    for obslits in mbobs:
        k = 0
        for obs in obslits:
            pixels_list.append(obs.pixels)
            psf_obs = obs.psf
            if psf_obs.has_gmix():
                psf_pars = psf_obs.gmix.get_full_pars()
                psf_moments.append(psf_pars[3:6])
                idx += 1
            else:
                raise ValueError("PSF has no gmix set.")
            k += 1
        band_tracker.append(k)
    psf_moments = np.array(psf_moments)
    band_tracker = np.array(band_tracker)

    tmparray = tmp_func(sum(band_tracker))

    # Gal
    find_ellipmom2(
        pixels_list,
        band_tracker,
        guess,
        resarray,
        tmparray,
        confarray,
        psf_moments=psf_moments,
    )

    x0_f, y0_f, yy_f, xy_f, xx_f = resarray[0]["pars"][:5]

    # if abs(xx_f) < 1e-3:
    #     xx_f = 1e-3
    # if abs(yy_f) < 1e-3:
    #     yy_f = 1e-3
    # # if abs(xy_f) < 1e-3:
    # #     xy_f = 0.0
    if resarray[0]["flags"] != 0:
        return

    flux_gal = (
        resarray[0]["pars"][6:]
        / resarray[0]["wnorm"]
        / mbobs[0][0].jacobian.area
    )

    # Correct for PSF residuals
    pixels_list_resi = []
    k = 0
    for nb, obslits in enumerate(mbobs):
        for obs in obslits:
            psf_pars = obs.psf.gmix.get_full_pars()
            if "psf_resi" not in obs.meta.keys():
                resi_img = get_resi_img(
                    obs,
                    xx_f,
                    xy_f,
                    yy_f,
                    flux_gal[nb],
                    psf_pars[5],
                    psf_pars[4],
                    psf_pars[3],
                    psf_pars[0],
                )
            else:
                resi_img = get_true_resi_img(
                    obs,
                    x0_f,
                    y0_f,
                    xx_f,
                    yy_f,
                    xy_f,
                    flux_gal[nb],
                    obs.meta["psf_resi"],
                )
            resi_obs = obs.copy()
            resi_obs.set_image(resi_img, update_pixels=True)
            pixels_list_resi.append(resi_obs.pixels)
            k += 1

    # Get resi
    guess_resi = np.array(
        [
            resarray[0]["pars"][0],
            resarray[0]["pars"][1],
            yy_f,
            xy_f,
            xx_f,
        ]
    )

    find_ellipmom2_bj(
        pixels_list_resi,
        band_tracker,
        guess_resi,
        resarray,
        tmparray,
        confarray,
        psf_moments=psf_moments,
    )
    xx_final, xy_final, yy_final = resarray[0]["pars"][2:5]

    if np.isnan(xx_final) or np.isnan(yy_final) or np.isnan(xy_final):
        # NOTE: Probably not the best flags
        resarray[0]["flags"] = ngmix.flags.LOW_DET
        return

    # NOTE: I think this is not technically correct as the PSF uncertainties
    # should be included. We could also consider that they are really small and
    # that we are dominated by the Galaxy measurement errors.
    # NOTE 2: We are also neglecting the ellipticity manipulation from the
    # `bj_nullPSF` and `get_corrected_mom3` functions.
