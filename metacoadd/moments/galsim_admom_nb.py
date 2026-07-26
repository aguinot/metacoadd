"""
This an implementation of the Galsim adaptive moments algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp
"""

from math import atan2, cos, exp, sin, sqrt

import ngmix
import numpy as np
from numba import njit


@njit(cache=True)
def find_ellipmom1(
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
    do_cov=False,
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
        detM = Mxx * Myy - Mxy * Mxy
        w_norm = 1.0 / (2 * np.pi * sqrt(detM))
        res["wnorm"] = 1.0

        Minv_xx = Myy / detM
        TwoMinv_xy = -Mxy / detM * 2.0
        Minv_yy = Mxx / detM

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
            if tmp_sums[6 + band_ind] <= 0.0:
                res["flags"] = ngmix.flags.NONPOS_FLUX
                return
            tmp_sums[:6] /= tmp_sums[6 + band_ind]
            res["sums"][:] += tmp_sums * ivar_sum
            flux_weights[band_ind] += ivar_sum
        else:
            flux = tmp["sums"][i_list][6]
            if flux <= 0.0:
                res["flags"] = ngmix.flags.NONPOS_FLUX
                return
            normalize_moment_covariance(
                tmp["sums"][i_list], tmp["sums_cov"][i_list]
            )
            tmp["sums"][i_list][6] /= w_norm
        tracking += 1
        if tracking == band_tracker[band_ind]:
            band_ind += 1
            tracking = 0

    if do_cov:
        combine_multiband_observations_array(res, tmp, band_tracker)
        res["s2n"] = sqrt(
            res["sums"] @ np.linalg.solve(res["sums_cov"], res["sums"])
        )
    else:
        res["sums"][:] /= np.sum(flux_weights)

    res["wsum"] /= n_list


@njit(cache=True)
def normalize_moment_covariance(sums, sums_cov):
    """Transform raw weighted sums and covariance to flux-normalized moments."""
    raw_cov = sums_cov.copy()
    flux = sums[6]
    jac = np.zeros((7, 7), dtype=np.float64)

    for i in range(6):
        jac[i, i] = 1.0 / flux
        jac[i, 6] = -sums[i] / (flux * flux)
    jac[6, 6] = 1.0

    sums_cov[:, :] = jac @ raw_cov @ jac.T
    sums[:6] /= flux


@njit(cache=True)
def find_ellipmom2(
    pixels_list,
    band_tracker,
    guess,
    resarray,
    tmparray,
    confarray,
):
    """ """

    conf = confarray[0]
    res = resarray[0]
    tmp = tmparray[0]

    convergence_factor = 1.0
    shiftscale0 = 0.0

    pix_scale = np.sqrt(np.mean(pixels_list[0]["area"]))

    x0, y0, Mxx, Mxy, Myy = guess
    x00 = x0
    y00 = y0
    do_cov = False

    for i in range(conf["maxiter"]):
        clear_result(res)
        clear_tmp(tmp)
        find_ellipmom1(
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
        )
        if res["flags"] != 0:
            return
        Bx, By, Cxx, Cxy, Cyy, rho4 = res["sums"][:6]
        Amps = res["sums"][6:]
        if do_cov:
            Amp, _, _, _ = compute_effective_flux(
                Amps,
                res["sums_cov"][6:, 6:],
            )
        else:
            Amp = sum(Amps)

        if Amp <= 0:
            res["flags"] = ngmix.flags.NONPOS_FLUX
            return

        two_psi = atan2(2 * Mxy, Mxx - Myy)
        semi_a2 = 0.5 * ((Mxx + Myy) + (Mxx - Myy) * cos(two_psi)) + (
            Mxy * sin(two_psi)
        )
        semi_b2 = Mxx + Myy - semi_a2

        if semi_b2 <= 0:
            res["flags"] = ngmix.flags.NONPOS_SIZE
            return

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
        if dy > conf["bound_correct_wt"]:
            dy = conf["bound_correct_wt"]
        if dy < -conf["bound_correct_wt"]:
            dy = -conf["bound_correct_wt"]
        if dxx > conf["bound_correct_wt"]:
            dxx = conf["bound_correct_wt"]
        if dxx < -conf["bound_correct_wt"]:
            dxx = -conf["bound_correct_wt"]
        if dxy > conf["bound_correct_wt"]:
            dxy = conf["bound_correct_wt"]
        if dxy < -conf["bound_correct_wt"]:
            dxy = -conf["bound_correct_wt"]
        if dyy > conf["bound_correct_wt"]:
            dyy = conf["bound_correct_wt"]
        if dyy < -conf["bound_correct_wt"]:
            dyy = -conf["bound_correct_wt"]

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
            return

        res["numiter"] = i + 1

        if convergence_factor < conf["tol"]:  # or do_cov:
            if not do_cov:
                do_cov = True
                continue
            # rho4 /= Amp
            res["pars"][0] = x0
            res["pars"][1] = y0
            res["pars"][2] = Mxx
            res["pars"][3] = Mxy
            res["pars"][4] = Myy
            res["pars"][5] = rho4
            res["pars"][6:] = Amps * rho4
            break

        if res["numiter"] == conf["maxiter"]:
            res["flags"] = ngmix.flags.MAXITER


@njit(cache=True)
def clear_result(res):
    """
    clear some fields in the result structure
    """
    res["npix"] = 0
    res["wsum"] = 0.0
    res["sums"][:] = 0.0
    res["sums_cov"][:, :] = 0.0
    res["pars"][:] = np.nan


@njit(cache=True)
def clear_tmp(tmp):
    """
    clear some fields in the result structure
    """
    tmp["sums"][:] = 0.0
    tmp["sums_cov"][:, :] = 0.0


@njit(cache=True)
def compute_effective_flux(fluxes, flux_cov):
    """
    Compute effective flux, its variance, and optional cross-covariances
    with other parameters using the BLUE estimator.

    Parameters
    ----------
    fluxes : (B,) array
        The flux vector (e.g. from multiple bands).

    flux_cov : (B, B) array
        Covariance matrix of the flux vector.

    Returns
    -------
    F_eff : float
        Effective flux via BLUE estimator.

    F_eff_var : float
        Variance of the effective flux.

    weights : (B,) array
        BLUE weights applied to the input fluxes.

    flux_vars : (B,) array
        Marginal variances of the input fluxes.
    """
    ones = np.ones(len(fluxes), dtype=np.float64)

    # Solve flux_cov @ x = ones
    flux_weights = np.linalg.solve(flux_cov, ones)

    denom = ones @ flux_weights
    weights = flux_weights / denom

    F_eff = weights @ fluxes
    F_eff_var = 1.0 / denom
    return F_eff, F_eff_var, weights, np.diag(flux_cov)


@njit(cache=True)
def compute_flux_cross_covs(flux_weights, target_covs):
    """
    Compute effective flux, its variance, and optional cross-covariances
    with other parameters using the BLUE estimator.

    Parameters
    ----------

    target_covs : (N, B) array, optional
        Covariances between each of N target parameters and the fluxes.
        If provided, returns cross-covariances with F_eff.

    Returns
    -------
    cross_covs : (N,) array, optional
        Cross-covariances with F_eff, if target_covs was provided.
    """
    target_covs = np.atleast_2d(target_covs)
    cross_covs = target_covs @ flux_weights

    return cross_covs


@njit(cache=True)
def combine_multiband_observations_array(res, tmp, band_tracker):
    """
    Combine multi-band measurements from array inputs.

    Parameters:
    -----------
    m_array : ndarray, shape (N_obs, 7)
        [x0, y0, Q11, Q12, Q22, rho2, F] for each observation

    Sigma_array : ndarray, shape (N_obs, 7, 7)
        Covariance matrices for each observation

    band_tracker : list of int
        List of number of observations per band. Length = N_bands

    Returns:
    --------
    Q_joint : np.ndarray, shape (3,)
        Joint shape [Q11, Q22, Q12]

    F_b : np.ndarray, shape (N_bands,)
        Combined flux per band

    Sigma_M : np.ndarray, shape (6 + N_bands, 6 + N_bands)
        Full covariance matrix for [x0, y0, Q11, Q12, Q22, rho2, F_1, ..., F_B]

    x0_joint, y0_joint : float
        Joint centroid
    """

    # Unpack the input
    m_array = tmp["sums"]
    Sigma_array = tmp["sums_cov"]
    N_obs = m_array.shape[0]
    N_bands = len(band_tracker)

    # === Step 1: Combine shared parameters ===
    sum_inv_Sigma_shared = np.zeros((6, 6))
    sum_inv_Sigma_shared_x = np.zeros(6)

    for j in range(N_obs):
        x_shared = m_array[j, :6]
        Sigma_shared = Sigma_array[j, :6, :6]
        inv_Sigma_shared = np.linalg.inv(Sigma_shared)
        sum_inv_Sigma_shared += inv_Sigma_shared
        sum_inv_Sigma_shared_x += inv_Sigma_shared @ x_shared

    Sigma_shared_joint = np.linalg.inv(sum_inv_Sigma_shared)
    x_shared_joint = Sigma_shared_joint @ sum_inv_Sigma_shared_x

    # === Step 2: Combine per-band fluxes ===
    F_b = np.zeros(N_bands)
    flux_weights = np.zeros(sum(band_tracker), dtype=np.float64)

    idx = 0
    for b in range(N_bands):
        n_obs_b = band_tracker[b]
        sum_inv_var = 0.0
        sum_inv_var_flux = 0.0

        for e in range(n_obs_b):
            F = m_array[idx, -1]
            var_F = Sigma_array[idx, -1, -1]
            inv_var = 1.0 / var_F

            sum_inv_var += inv_var
            sum_inv_var_flux += inv_var * F
            flux_weights[idx] = inv_var
            idx += 1

        # for i in flux_weights[b]:
        start_idx = idx - n_obs_b
        flux_weights[start_idx:idx] /= sum_inv_var

        F_b[b] = sum_inv_var_flux / sum_inv_var

    # === Step 3: Full covariance matrix ===
    M_dim = 6 + N_bands
    Sigma_M = np.zeros((M_dim, M_dim))

    start = 0
    for b in range(N_bands):
        n_obs_b = band_tracker[b]
        for i in range(n_obs_b):
            idx = start + i
            Sigma_bi = Sigma_array[idx]

            W_bi = np.zeros((M_dim, 7))

            # Shared part
            Sigma_shared_bi = Sigma_bi[:6, :6]
            inv_Sigma_shared_bi = np.linalg.inv(Sigma_shared_bi)
            shared_weight = inv_Sigma_shared_bi @ Sigma_shared_joint
            W_bi[:6, :6] = shared_weight.T

            # Flux part
            w_flux = flux_weights[idx]
            W_bi[6 + b, -1] = w_flux

            # Propagate
            Sigma_M += W_bi @ Sigma_bi @ W_bi.T

        start += n_obs_b

    res["sums"][:6] = x_shared_joint
    res["sums"][6:] = F_b
    res["sums_cov"] = Sigma_M


@njit(cache=True)
def get_mom_var(
    X, Y, Z, var_X, var_Y, var_Z, var_XY, var_XZ, var_YZ, kind="e1"
):
    dfdx = dfdy = dfdz = 0
    T = X + Y
    if kind == "e1":
        dfdx = 2 * Y / T**2
        dfdy = -2 * X / T**2
    elif kind == "e2":
        dfdx = dfdy = -2 * Z / T**2
        dfdz = 2 / T
    elif kind == "T":
        dfdx = 1 / Z
        dfdy = 1 / Z
        dfdz = -(X + Y) / Z**2

    var_t = (
        dfdx**2 * var_X
        + dfdy**2 * var_Y
        + dfdz**2 * var_Z
        + 2 * dfdx * dfdy * var_XY
        + 2.0 * dfdx * dfdz * var_XZ
        + 2.0 * dfdy * dfdz * var_YZ
    )

    return var_t


@njit(cache=True)
def get_flux_var(flux, rho4, var_flux, var_rho4, cov_flux_rho4):
    """Propagate the variance of the product ``flux * rho4``."""
    return (
        flux * flux * var_rho4
        + rho4 * rho4 * var_flux
        + 2.0 * flux * rho4 * cov_flux_rho4
    )
