"""This an implementation of the Galsim adaptive moments algorithm in ngmix
format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp.
"""

import os
from math import atan2, cos, exp, sin, sqrt

import ngmix
import numpy as np
import numba as nb
from numba import njit

# This allows having coverage
if (
    os.environ.get("TESTING_GALSIM_ADMOM", "0") == "1"
    and os.environ.get("COVERAGE_MODE", "0") == "1"
):
    nb.config.DISABLE_JIT = 1


@njit(cache=True)
def find_ellipmom1(
    pixels_list,
    values_list,
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
    """Compute the weighted sums of moments for a given guess of the centroid
    and second moments. This is the first step of the adaptive moment
    algorithm.

    Parameters
    ----------
    pixels_list : list[np.ndarray]
        Each structured array contains the pixel data for one observation. The
        fields should include 'u', 'v', 'area', and 'ierr'.
    values_list : list[np.ndarray]
        Each array contains the pixel values for one observation.
    band_tracker : list[int]
        A list of integers that tracks the number of observations in each band.
    x0 : float
        The x-coordinate of the initial guess for the centroid.
    y0 : float
        The y-coordinate of the initial guess for the centroid.
    Mxx : float
        The xx moment of the initial guess for the second moments.
    Mxy : float
        The xy moment of the initial guess for the second moments.
    Myy : float
        The yy moment of the initial guess for the second moments.
    res : dict
        A dictionary to store the results of the computation.
    tmp : dict
        A dictionary to store temporary values during the computation.
    conf : dict
        A dictionary containing configuration parameters.
    do_cov : bool, optional
        If True, compute the covariance matrix. Default is False.

    """
    F = res["F"]

    if not do_cov:
        flux_weights = tmp["flux_weights"]
        flux_weights[:] = 0.0
        tmp["norm_weights"][:] = 0.0
        tmp["raw_flux"][:] = 0.0

    tmp_sums = tmp["tmp_sums"]

    detM = Mxx * Myy - Mxy * Mxy
    w_norm = 1.0 / (2 * np.pi * sqrt(detM))
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    res["wnorm"] = 1.0

    tracking = 0
    band_ind = 0
    n_list = len(pixels_list)
    for i_list in range(n_list):
        pixels = pixels_list[i_list]
        values = values_list[i_list]
        tmp_sums[:] = 0.0
        ivar_sum = 0.0
        n_pixels = pixels.size
        for i_pix in range(n_pixels):
            pixel = pixels[i_pix]

            umod = pixel["u"] - x0
            vmod = pixel["v"] - y0

            vvmod = vmod * vmod
            uvmod = umod * vmod
            uumod = umod * umod

            Minv_xx__x_x0__x_x0 = Minv_xx * vvmod
            TwoMinv_xy__y_y0__x_x0 = TwoMinv_xy * uvmod
            Minv_yy__y_y0__y_y0 = Minv_yy * uumod

            rho2 = (
                Minv_yy__y_y0__y_y0
                + TwoMinv_xy__y_y0__x_x0
                + Minv_xx__x_x0__x_x0
            )

            res["npix"] += 1
            if rho2 < conf["max_moment_nsig2"]:
                win = exp(-0.5 * rho2) * w_norm * pixel["area"]
                intensity = win * values[i_pix]

                ivar = pixel["ierr"] * pixel["ierr"]

                res["wsum"] += win
                ivar_sum += ivar

                if not do_cov:
                    tmp_sums[0] += umod * intensity
                    tmp_sums[1] += vmod * intensity
                    tmp_sums[2] += vvmod * intensity
                    tmp_sums[3] += uvmod * intensity
                    tmp_sums[4] += uumod * intensity
                    tmp_sums[5] += rho2 * rho2 * intensity
                    tmp_sums[6 + band_ind] += 1.0 * intensity
                else:
                    win2 = win * win
                    var = 1.0 / ivar
                    F[0] = umod
                    F[1] = vmod
                    F[2] = vvmod
                    F[3] = uvmod
                    F[4] = uumod
                    F[5] = rho2 * rho2
                    F[6] = 1.0
                    for i in range(7):
                        tmp["sums"][i_list][i] += intensity * F[i]
                        for j in range(i, 7):
                            tmp["sums_cov"][i_list][i, j] += (
                                win2 * var * F[i] * F[j]
                            )

        if not do_cov:
            if tmp_sums[6 + band_ind] <= 0.0:
                res["flags"] = ngmix.flags.NONPOS_FLUX
                return
            tmp["raw_flux"][i_list] = tmp_sums[6 + band_ind] / (
                w_norm * pixels[0]["area"]
            )
            tmp_sums[:6] /= tmp_sums[6 + band_ind]
            res["sums"][:] += tmp_sums * ivar_sum
            flux_weights[band_ind] += ivar_sum
            tmp["norm_weights"][i_list] = ivar_sum
        else:
            for i in range(7):
                for j in range(i + 1, 7):
                    tmp["sums_cov"][i_list][j, i] = tmp["sums_cov"][i_list][
                        i, j
                    ]
            flux = tmp["sums"][i_list][6]
            if flux <= 0.0:
                res["flags"] = ngmix.flags.NONPOS_FLUX
                return
            normalize_moment_covariance(
                tmp["sums"][i_list],
                tmp["sums_cov"][i_list],
                flux_scale=1.0 / w_norm,
            )
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
        tmp["norm_weights"][:] /= np.sum(tmp["norm_weights"])

    res["wsum"] /= n_list


@njit(cache=True)
def normalize_moment_covariance(
    sums,
    sums_cov,
    flux_scale,
):
    """Normalize moments by flux and rescale the flux statistic.

    Parameters
    ----------
    sums : np.ndarray
        Raw weighted moments followed by the weighted flux.
    sums_cov : np.ndarray
        Covariance of the raw weighted moments and flux.
    flux_scale : float
        Multiplicative conversion applied to the weighted flux.

    """
    raw_cov = sums_cov.copy()
    flux = sums[6]

    jac = np.zeros((7, 7), dtype=np.float64)

    # S_i -> S_i / flux
    for i in range(6):
        jac[i, i] = 1.0 / flux
        jac[i, 6] = -sums[i] / flux**2

    # flux -> flux * flux_scale
    jac[6, 6] = flux_scale

    sums_cov[:, :] = jac @ raw_cov @ jac.T
    sums[:6] /= flux
    sums[6] *= flux_scale


@njit(cache=True)
def find_ellipmom2(
    pixels_list,
    values_list,
    band_tracker,
    guess,
    resarray,
    tmparray,
    confarray,
    do_covariance=True,
):
    """Compute the adaptive moments for a set of observations.

    Parameters
    ----------
    pixels_list : list[np.ndarray]
        Each structured array contains the pixel data for one observation. The
        fields should include 'u', 'v', 'area', and 'ierr'.
    values_list : list[np.ndarray]
        Each array contains the pixel values for one observation.
    band_tracker : list[int]
        A list of integers indicating which band each observation belongs to.
    guess : tuple of floats
        Initial guess for the centroid and second moments
        (x0, y0, Mxx, Mxy, Myy).
    resarray : np.ndarray, shape (1,)
        Array to store the results of the computation.
    tmparray : np.ndarray, shape (1,)
        Array to store temporary values during the computation.
    confarray : np.ndarray, shape (1,)
        Array containing configuration parameters for the computation.
    do_covariance : bool, optional
        If True, compute the covariance matrix of the adaptive moments.
        Default is True.

    """
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
        clear_result(res, clear_covariance=do_cov)
        clear_tmp(tmp, clear_covariance=do_cov)
        find_ellipmom1(
            pixels_list,
            values_list,
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

        convergence_factor = max(
            abs(dx),
            abs(dy),
            abs(dxx),
            abs(dxy),
            abs(dyy),
        )
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
            if not do_cov and do_covariance:
                do_cov = True
                continue

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
def clear_result(res, clear_covariance=True):
    """Clear some fields in the result structure.

    Parameters
    ----------
    res : dict
        The result structure to clear.
    clear_covariance : bool, optional
        If True, clear the covariance matrix. Default is True.

    """
    res["npix"] = 0
    res["wsum"] = 0.0
    res["sums"][:] = 0.0
    if clear_covariance:
        res["sums_cov"][:, :] = 0.0
    res["pars"][:] = np.nan


@njit(cache=True)
def clear_tmp(tmp, clear_covariance=True):
    """Clear some fields in the temporary structure.

    Parameters
    ----------
    tmp : dict
        The temporary structure to clear.
    clear_covariance : bool, optional
        If True, clear the covariance matrix. Default is True.

    """
    if clear_covariance:
        tmp["sums"][:] = 0.0
        tmp["sums_cov"][:, :] = 0.0


@njit(cache=True)
def compute_effective_flux(fluxes, flux_cov):
    """Compute effective flux, its variance, and optional cross-covariances
    with other parameters using the BLUE estimator.

    Parameters
    ----------
    fluxes : (B,) np.ndarray
        The flux vector (e.g. from multiple bands).

    flux_cov : (B, B) np.ndarray
        Covariance matrix of the flux vector.

    Returns
    -------
    F_eff : float
        Effective flux via BLUE estimator.

    F_eff_var : float
        Variance of the effective flux.

    weights : (B,) np.ndarray
        BLUE weights applied to the input fluxes.

    flux_vars : (B,) np.ndarray
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
    """Compute effective flux, its variance, and optional cross-covariances
    with other parameters using the BLUE estimator.

    Parameters
    ----------
    flux_weights : np.ndarray
        BLUE weights applied to the input fluxes.
    target_covs : np.ndarray
        Covariances between each of N target parameters and the fluxes.
        If provided, returns cross-covariances with F_eff.

    Returns
    -------
    cross_covs : np.ndarray
        Cross-covariances with F_eff, if target_covs was provided.

    """
    target_covs = np.atleast_2d(target_covs)
    cross_covs = target_covs @ flux_weights

    return cross_covs


@njit(cache=True)
def combine_multiband_observations_array(res, tmp, band_tracker):
    """Combine multi-band measurements from array inputs.

    Parameters
    ----------
    res : dict
        The result structure to store the combined observations.
    tmp : dict
        The temporary structure to store intermediate results.
    band_tracker : list of int
        List of number of observations per band. Length = N_bands

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

        for _ in range(n_obs_b):
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
    """Propagate fixed-weight moment covariance to adaptive size and
    ellipticity.

    Parameters
    ----------
    X : float
        Row-row second moment.
    Y : float
        Column-column second moment.
    Z : float
        Row-column second moment.
    var_X : float
        Variance of ``X``.
    var_Y : float
        Variance of ``Y``.
    var_Z : float
        Variance of ``Z``.
    var_XY : float
        Covariance between ``X`` and ``Y``.
    var_XZ : float
        Covariance between ``X`` and ``Z``.
    var_YZ : float
        Covariance between ``Y`` and ``Z``.
    kind : str, optional
        The type of moment to compute the variance for.
        Options are "e1", "e2", or "T". Default is "e1".

    Returns
    -------
    var_t : float
        Variance of the specified moment type after adaptive-response scaling.

    """
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
def get_T_and_e_cov(
    Q22,
    Q11,
    Q12,
    var_Q22,
    var_Q11,
    var_Q12,
    cov_Q22_Q11,
    cov_Q22_Q12,
    cov_Q11_Q12,
):
    """Propagate fixed-weight moment covariance to adaptive size and
    ellipticity.

    Parameters are the flux-normalized second moments and their covariance in
    ngmix/image-coordinate ordering: ``Q22`` is the row-row moment, ``Q11`` is
    the column-column moment, and ``Q12`` is the row-column cross moment. The
    reported ellipticity convention is
    ``e1 = (Q11 - Q22) / (Q11 + Q22)`` and
    ``e2 = 2 * Q12 / (Q11 + Q22)``.

    The input covariance describes moments measured with the final fixed
    Gaussian weight. At the adaptive-moment fixed point the fitted covariance
    matrix is ``M = 2 C``, so first-order perturbations in the fitted moments
    include an additional adaptive-response factor relative to fixed-weight
    moment errors. The returned variances include this response factor.

    Parameters
    ----------
    Q22 : float
        Row-row second moment.
    Q11 : float
        Column-column second moment.
    Q12 : float
        Row-column second moment.
    var_Q22 : float
        Variance of ``Q22``.
    var_Q11 : float
        Variance of ``Q11``.
    var_Q12 : float
        Variance of ``Q12``.
    cov_Q22_Q11 : float
        Covariance between ``Q22`` and ``Q11``.
    cov_Q22_Q12 : float
        Covariance between ``Q22`` and ``Q12``.
    cov_Q11_Q12 : float
        Covariance between ``Q11`` and ``Q12``.

    Returns
    -------
    T_var : float
        Variance of ``T = Q11 + Q22`` after adaptive-response scaling.
    e1_var : float
        Variance of ``e1`` after adaptive-response scaling.
    e2_var : float
        Variance of ``e2`` after adaptive-response scaling.
    e12_cov : float
        Covariance between ``e1`` and ``e2`` after adaptive-response scaling.

    """
    T = Q22 + Q11
    inv_T = 1.0 / T
    inv_T2 = inv_T * inv_T

    de1_dQ22 = 2.0 * Q11 * inv_T2
    de1_dQ11 = -2.0 * Q22 * inv_T2
    de2_dQ22 = -2.0 * Q12 * inv_T2
    de2_dQ11 = de2_dQ22
    de2_dQ12 = 2.0 * inv_T

    fixed_e1_var = (
        de1_dQ22 * de1_dQ22 * var_Q22
        + de1_dQ11 * de1_dQ11 * var_Q11
        + 2.0 * de1_dQ22 * de1_dQ11 * cov_Q22_Q11
    )
    fixed_e2_var = (
        de2_dQ22 * de2_dQ22 * var_Q22
        + de2_dQ11 * de2_dQ11 * var_Q11
        + de2_dQ12 * de2_dQ12 * var_Q12
        + 2.0 * de2_dQ22 * de2_dQ11 * cov_Q22_Q11
        + 2.0 * de2_dQ22 * de2_dQ12 * cov_Q22_Q12
        + 2.0 * de2_dQ11 * de2_dQ12 * cov_Q11_Q12
    )
    fixed_e12_cov = (
        de1_dQ22 * de2_dQ22 * var_Q22
        + de1_dQ11 * de2_dQ11 * var_Q11
        + (de1_dQ22 * de2_dQ11 + de1_dQ11 * de2_dQ22) * cov_Q22_Q11
        + de1_dQ22 * de2_dQ12 * cov_Q22_Q12
        + de1_dQ11 * de2_dQ12 * cov_Q11_Q12
    )

    # At the adaptive fixed point M = 2 C. Noise perturbs the fitted
    # covariance by delta M = 2 delta C to first order, so adaptive-moment
    # covariance is four times the fixed-weight covariance.
    adaptive_response2 = 4.0
    # The fitted covariance is M = 2 C, so its trace contributes another
    # factor two in standard deviation in addition to adaptive response.
    T_var = 4.0 * adaptive_response2 * (var_Q22 + var_Q11 + 2.0 * cov_Q22_Q11)
    e1_var = adaptive_response2 * fixed_e1_var
    e2_var = adaptive_response2 * fixed_e2_var
    e12_cov = adaptive_response2 * fixed_e12_cov

    return T_var, e1_var, e2_var, e12_cov


@njit(cache=True)
def get_flux_var(flux, rho4, var_flux, var_rho4, cov_flux_rho4):
    """Propagate the variance of the product ``flux * rho4``.

    Parameters
    ----------
    flux : float
        The flux value.
    rho4 : float
        The fourth moment of the radial profile.
    var_flux : float
        Variance of the flux.
    var_rho4 : float
        Variance of the fourth moment.
    cov_flux_rho4 : float
        Covariance between the flux and the fourth moment.

    Returns
    -------
    var_flux_rho4 : float
        Variance of the product ``flux * rho4``.

    """
    return (
        flux * flux * var_rho4
        + rho4 * rho4 * var_flux
        + 2.0 * flux * rho4 * cov_flux_rho4
    )
