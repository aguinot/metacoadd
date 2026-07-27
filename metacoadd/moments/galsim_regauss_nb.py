"""
This an implementation of the Galsim re-gauss algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp

This implementation is modified to allowed post-metacalibration PSF correction.
"""

from math import exp, sqrt, floor, ceil, log

import numpy as np
from numba import njit
from numba.typed import List

import ngmix
from ngmix.moments import mom2e

from .galsim_admom_nb import find_ellipmom2


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

    # NumPy FFT calls are compiled by rocket-fft's Numba overloads.
    kim1 = np.fft.rfft2(xim, s=(N, N))

    # Do fft img2
    xim.fill(0)
    b2 = [
        offset,
        image2.shape[0] + offset - 1,
        offset,
        image2.shape[1] + offset - 1,
    ]
    xim[b2[0] : b2[1] + 1, b2[2] : b2[3] + 1] = image2
    kim2 = np.fft.rfft2(xim, s=(N, N))

    # Conv
    kim2 *= kim1

    # Inverse fft
    xim = np.fft.irfft2(kim2, s=(N, N))

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
    half_n = N // 2
    for row in range(b3[0], b3[1]):
        shifted_row = (row + half_n) % N
        out_row = row + shift_x
        for col in range(b3[2], b3[3]):
            shifted_col = (col + half_n) % N
            out_col = col + shift_y
            image_out[out_row, out_col] += xim[shifted_row, shifted_col]


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

    jac = obs.jacobian
    psf_obs = obs.psf
    psf_jac = psf_obs.jacobian
    jac_area = jac.area

    x_gal_min, y_gal_min = 0, 0
    x_gal_max, y_gal_max = obs.image.shape
    x_psf_min, y_psf_min = 0, 0
    x_psf_max, y_psf_max = psf_obs.image.shape

    if xx_f <= jac_area:
        xx_f = jac_area
    if yy_f <= jac_area:
        yy_f = jac_area

    # Get fgauss bounds
    fgauss_xmin = x_gal_min - x_psf_max
    fgauss_xmax = x_gal_max - x_psf_min
    fgauss_ymin = y_gal_min - y_psf_max
    fgauss_ymax = y_gal_max - y_psf_min
    fgauss_xctr = jac.row0 - psf_jac.row0
    fgauss_yctr = jac.col0 - psf_jac.col0
    fgauss_xsig = np.sqrt(xx_f / jac_area)
    fgauss_ysig = np.sqrt(yy_f / jac_area)
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
    f_jac = ngmix.Jacobian(row=f_row0, col=f_col0, wcs=jac.get_galsim_wcs())

    # Get PSF bounds
    p_xmin = int(
        floor(
            psf_jac.row0
            - nsig_rg2 * sqrt(xx_f / jac_area)
            - nsig_rg * fgauss_xsig
        )
    )
    p_xmax = int(
        ceil(
            psf_jac.row0
            + nsig_rg2 * sqrt(xx_f / jac_area)
            + nsig_rg * fgauss_xsig
        )
    )
    p_ymin = int(
        floor(
            psf_jac.col0
            - nsig_rg2 * sqrt(yy_f / jac_area)
            - nsig_rg * fgauss_ysig
        )
    )
    p_ymax = int(
        ceil(
            psf_jac.col0
            + nsig_rg2 * sqrt(yy_f / jac_area)
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
        row=p_row0, col=p_col0, wcs=psf_jac.get_galsim_wcs()
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
        -psf_obs.image[p_xmin : p_xmax + 1, p_ymin : p_ymax + 1] + fpsf_img
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
    do_covariance=True,
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
    pixels_list = List(pixels_list)
    values_list = List([pixels["val"] for pixels in pixels_list])

    tmparray = tmp_func(sum(band_tracker), len(band_tracker))

    # Gal
    find_ellipmom2(
        pixels_list,
        values_list,
        band_tracker,
        guess,
        resarray,
        tmparray,
        confarray,
        False,
    )

    psf_moments_eff = np.zeros(3, dtype=np.float64)
    for i in range(len(psf_moments)):
        psf_moments_eff += tmparray[0]["norm_weights"][i] * psf_moments[i]

    x0_f, y0_f, yy_g, xy_g, xx_g = resarray[0]["pars"][:5]
    yy_f = yy_g - psf_moments_eff[0]
    xy_f = xy_g - psf_moments_eff[1]
    xx_f = xx_g - psf_moments_eff[2]

    if resarray[0]["flags"] != 0:
        return
    if (
        (not np.isfinite(xx_f))
        or (not np.isfinite(xy_f))
        or (not np.isfinite(yy_f))
        or xx_f <= 0.0
        or yy_f <= 0.0
        or xx_f * yy_f - xy_f * xy_f <= 0.0
    ):
        resarray[0]["flags"] = ngmix.flags.NONPOS_SIZE
        return

    flux_gal = np.zeros(len(band_tracker), dtype=np.float64)
    obs_index = 0
    for band in range(len(band_tracker)):
        band_weight = 0.0
        band_amplitude = 0.0
        for _ in range(band_tracker[band]):
            weight = tmparray[0]["norm_weights"][obs_index]
            band_weight += weight
            band_amplitude += weight * tmparray[0]["raw_flux"][obs_index]
            obs_index += 1
        flux_gal[band] = resarray[0]["pars"][5] * band_amplitude / band_weight

    # Correct for PSF residuals
    values_list_resi = List()
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
            values_list_resi.append(
                np.ascontiguousarray(resi_img[obs.weight > 0.0])
            )
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

    find_ellipmom2(
        pixels_list,
        values_list_resi,
        band_tracker,
        guess_resi,
        resarray,
        tmparray,
        confarray,
        do_covariance,
    )
    if resarray[0]["flags"] != 0:
        return
    yy_resi, xy_resi, xx_resi = resarray[0]["pars"][2:5]

    e1_psf, e2_psf, T_psf = mom2e(
        psf_moments_eff[0], psf_moments_eff[1], psf_moments_eff[2]
    )
    xx_final, xy_final, yy_final = BJ_correction(
        yy_resi,
        xy_resi,
        xx_resi,
        resarray[0]["pars"][5],
        e1_psf,
        e2_psf,
        T_psf,
    )

    if (
        (not np.isfinite(xx_final))
        or (not np.isfinite(xy_final))
        or (not np.isfinite(yy_final))
        or xx_final <= 0.0
        or yy_final <= 0.0
        or xx_final * yy_final - xy_final * xy_final <= 0.0
    ):
        # NOTE: Probably not the best flags
        resarray[0]["flags"] = ngmix.flags.LOW_DET
        return

    # Result moments use ngmix's (row-row, row-col, col-col) ordering.
    resarray[0]["pars"][2] = yy_final
    resarray[0]["pars"][3] = xy_final
    resarray[0]["pars"][4] = xx_final

    # NOTE: I think this is not technically correct as the PSF uncertainties
    # should be included. We could also consider that they are really small and
    # that we are dominated by the Galaxy measurement errors.
    # NOTE 2: We are also neglecting the ellipticity manipulation from the
    # `bj_nullPSF` and `get_corrected_mom3` functions.
