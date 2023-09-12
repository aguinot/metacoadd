"""
This is an emplementation of the re-gausianization method for PSF correction
addapted to the metacalibration case where the PSF is an isotropic gaussian.
"""

import numpy as np
from ngmix.gmix import GMixModel
from ngmix.gmix.gmix_nb import gmix_eval_pixel_fast, gmix_set_norms
from ngmix.moments import get_Tround, mom2g
from ngmix.shape import e1e2_to_g1g2
from numba import jit, njit

from .ngmix_admom import AdmomFitter


@jit
def get_rho4_ngmix(pixels, xx, yy, xy, det):
    inv_xx = yy / det
    inv_yy = xx / det
    inv_xy = -xy / det
    two_inv_xy = inv_xy * 2
    # inv_2_inv_xx = 0.5/inv_xx

    n_pixels = pixels.size

    rho4 = 0
    amp = 0
    for i_pix in range(n_pixels):
        pixel = pixels[i_pix]
        inv_xx_x_x0_x_x0 = inv_xx * pixel["u"] * pixel["u"]
        two_inv_xy_y_y0 = two_inv_xy * pixel["v"]
        inv_yy_y_y0_y_y0 = inv_yy * pixel["v"] * pixel["v"]

        rho2 = (
            inv_yy_y_y0_y_y0 + two_inv_xy_y_y0 * pixel["u"] + inv_xx_x_x0_x_x0
        )

        intensity = np.exp(-0.5 * rho2) * pixel["val"]

        amp += intensity
        rho4 += intensity * rho2 * rho2

    return rho4 / amp, rho4


@jit
def bj_nullPSF(T_ratio, e1_gal, e2_gal, rho4_gal):
    cosheta_g = 1 / np.sqrt(1 - e1_gal * e1_gal - e2_gal * e2_gal)
    sig2ratio = T_ratio * cosheta_g

    R = 1.0 - sig2ratio * (1 + rho4_gal) / (1 - rho4_gal) / cosheta_g

    return R


@jit
def get_admom(pars):
    xx = (pars[2] + pars[4]) / 2
    yy = (pars[4] - pars[2]) / 2
    xy = pars[3] / 2

    return xx, yy, xy


@jit
def get_corrected_mom(xx, yy, xy, R):
    a = xx - yy
    b = xx + yy
    c = 2 * xy
    a2 = a**2
    b2 = b**2
    c2 = c**2

    if (a.real < 0) & (b.real > a.real):
        x = -np.sqrt(a2 * (a2 - b2 + c2) / (R * (a2 - b2 * R + c2)))
        y = b * R * x / a
        z = c * x / a
    elif (a.real > 0) & (b.real > a.real):
        x = np.sqrt(a2 * (a2 - b2 + c2) / (R * (a2 - b2 * R + c2)))
        y = b * R * x / a
        z = c * x / a
    else:
        raise ValueError("Something went wrong...")

    xx_new = (x + y) / 2
    yy_new = (y - x) / 2
    xy_new = z / 2

    return xx_new, yy_new, xy_new


def get_psf_fit(obs, fitter, guess_fwhm=1.2, seed=None):
    res_psf = fitter.go(obs.psf, guess_fwhm)
    # xx_psf, yy_psf, xy_psf = get_admom(res_psf["pars"])
    xx_psf, xy_psf, yy_psf = res_psf["pars"][2:5]
    T_psf = xx_psf + yy_psf

    return xx_psf, yy_psf, xy_psf, T_psf


def check_exp(obs, psf_res, safe_factor=2):
    e1_psf = (psf_res[0] - psf_res[1]) / psf_res[3]
    e2_psf = (2 * psf_res[2]) / psf_res[3]
    g1_psf, g2_psf = e1e2_to_g1g2(e1_psf, e2_psf)
    pars = [0, 0, g1_psf, g2_psf, safe_factor * psf_res[3], 1.0]
    weight = GMixModel(pars, "gauss")
    w_data = weight._data
    gmix_set_norms(w_data)

    w_sum = _check_exp(obs.pixels, w_data)
    return w_sum


@njit
def _check_exp(pixels, w_data):
    w_sum = 0
    for pixel in pixels:
        val = gmix_eval_pixel_fast(w_data, pixel)
        w_sum += val
    return w_sum


def regauss(obs, psf_res, fitter=None, pars=None, do_fit=True, guess_fwhm=1.2):
    # Get PSF info
    xx_psf, yy_psf, xy_psf, T_psf = psf_res

    # Gal
    res_gal = fitter.go(obs, guess_fwhm)
    # xx_gal, yy_gal, xy_gal = get_admom(res_gal["pars"]/res_gal["wsum"])
    xx_gal, xy_gal, yy_gal = res_gal["pars"][2:5]
    det_gal = xx_gal * yy_gal - xy_gal * xy_gal

    T_gal = xx_gal + yy_gal
    e1_gal = (xx_gal - yy_gal) / T_gal
    e2_gal = (2 * xy_gal) / T_gal
    rho4gal, flux_gal = get_rho4_ngmix(
        obs.pixels, xx_gal, yy_gal, xy_gal, det_gal
    )

    R_bj = bj_nullPSF(T_psf / T_gal, e1_gal, e2_gal, 0.5 * rho4gal - 1)
    xx_final, yy_final, xy_final = get_corrected_mom(
        xx_gal,
        yy_gal,
        xy_gal,
        R_bj,
    )

    T_gal = (xx_gal - xx_psf) + (yy_gal - yy_psf)

    # e1_final = e1_gal/R_bj
    # e2_final = e2_gal/R_bj

    Res = T_gal / T_psf

    return flux_gal, T_gal, Res, xx_final, yy_final, xy_final, res_gal["wsum"]


def ME_regauss(obslist, guess_fwhm=1.2, seed=1234, safe_check=0.99):
    rng = np.random.RandomState(seed)
    fitter = AdmomFitter(rng=rng)

    n_epoch = len(obslist)
    seeds = rng.randint(0, 2**30, size=n_epoch)

    # First, fit PSF and check if exposures are good
    psf_res = []
    bad_check_sum = []
    check_sum = []
    for i, obs in enumerate(obslist):
        psf_res.append(get_psf_fit(obs, fitter, guess_fwhm, seed=seeds[i]))
        w_sum = check_exp(obs, psf_res[i])
        if w_sum < safe_check:
            bad_check_sum.append(i)
            check_sum.append(w_sum)

    if len(bad_check_sum) == len(obslist):
        raise ValueError("No good exposures found..")

    # Now measure good objects
    xx = 0
    yy = 0
    xy = 0
    flux = 0
    T = 0
    Res = 0
    norm = 0
    n_good = 0
    for i, obs in enumerate(obslist):
        if i in bad_check_sum:
            continue
        flux_tmp, T_tmp, Res_tmp, xx_tmp, yy_tmp, xy_tmp, w_sum_tmp = regauss(
            obs, psf_res[i], fitter=fitter, guess_fwhm=guess_fwhm, do_fit=True
        )
        xx += xx_tmp * w_sum_tmp
        yy += yy_tmp * w_sum_tmp
        xy += xy_tmp * w_sum_tmp
        flux += flux_tmp * w_sum_tmp
        T += T_tmp * w_sum_tmp
        Res += Res_tmp * w_sum_tmp
        norm += w_sum_tmp
        n_good += 1

    xx /= n_good
    yy /= n_good
    xy /= n_good
    flux /= n_good
    T /= n_good
    Res /= n_good

    g1, g2, _ = mom2g(yy, xy, xx)
    T = get_Tround(T, g1, g2)

    return g1, g2, T, flux, Res
