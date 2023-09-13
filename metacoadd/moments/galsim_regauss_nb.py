from math import ceil, exp, floor, log, sqrt

import ngmix
import numpy as np
import pyfftw
from numba import njit, objmode

from metacoadd.moments.galsim_admom_nb import find_ellipmom2


@njit(fastmath=True, cache=False)
def _check_exp(pixels, w_data):
    w_sum = 0
    for pixel in pixels:
        val = ngmix.gmix.gmix_nb.gmix_eval_pixel_fast(w_data, pixel)
        w_sum += val
    return w_sum


@njit(fastmath=True, cache=False)
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


@njit(fastmath=True, cache=False)
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


@njit(fastmath=True, cache=False)
def get_corrected_mom3(e1, e2, SB):
    """ """

    xx = 0.5 * np.sqrt(-((e1 + 1) ** 2) * SB**2 / (e1**2 + e2**2 - 1))
    yy = -xx * (e1 - 1) / (e1 + 1)
    xy = xx * e2 / (e1 + 1)

    return xx, yy, xy


@njit(fastmath=True, cache=False)
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


@njit(fastmath=True, cache=False)
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
    image_out[b4[0] : b4[1], b4[2] : b4[3]] += xim[b3[0] : b3[1], b3[2] : b3[3]]


def get_resi_img(
    obs, xx_gal, yy_gal, xy_gal, flux_gal, xx_psf, yy_psf, xy_psf, flux_psf
):
    nsig_rg = 3.0
    nsig_rg2 = 3.6

    x_gal_min, y_gal_min = 0, 0
    x_gal_max, y_gal_max = obs.image.shape
    x_psf_min, y_psf_min = 0, 0
    x_psf_max, y_psf_max = obs.psf.image.shape

    # Approx deconv
    xx_f = xx_gal - xx_psf
    yy_f = yy_gal - yy_psf
    xy_f = xy_gal - xy_psf
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
            - nsig_rg2 * sqrt(xx_gal / obs.jacobian.area)
            - nsig_rg * fgauss_xsig
        )
    )
    p_xmax = int(
        ceil(
            obs.psf.jacobian.row0
            + nsig_rg2 * sqrt(xx_gal / obs.jacobian.area)
            + nsig_rg * fgauss_xsig
        )
    )
    p_ymin = int(
        floor(
            obs.psf.jacobian.col0
            - nsig_rg2 * sqrt(yy_gal / obs.jacobian.area)
            - nsig_rg * fgauss_ysig
        )
    )
    p_ymax = int(
        ceil(
            obs.psf.jacobian.col0
            + nsig_rg2 * sqrt(yy_gal / obs.jacobian.area)
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

    g1_psf, g2_psf, T_psf = ngmix.moments.mom2g(yy_psf, xy_psf, xx_psf)
    pars_psf = np.zeros(6)
    pars_psf[2] = g1_psf
    pars_psf[3] = g2_psf
    pars_psf[4] = T_psf
    pars_psf[5] = flux_psf
    gmix_psf = ngmix.GMixModel(pars_psf, "gauss")

    fgauss_img = gmix_fgauss.make_image(
        (f_dim_x, f_dim_y), f_jac, fast_exp=True
    )
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


def regauss(obs, psf_res, fitter=None, pars=None, do_fit=True, guess_fwhm=0.6):
    # Get PSF info
    xx_psf, xy_psf, yy_psf = psf_res["pars"][2:5]
    T_psf = xx_psf + yy_psf

    e1_psf = (xx_psf - yy_psf) / T_psf
    e2_psf = 2.0 * xy_psf / T_psf

    flux_psf = psf_res["pars"][5] / psf_res["wnorm"] / obs.psf.jacobian.area

    # Gal
    res_gal = fitter._get_am_result()
    guess = fitter._generate_guess(obs, guess_fwhm)
    find_ellipmom2(obs.pixels, guess, res_gal, fitter.conf)
    xx_gal, xy_gal, yy_gal = res_gal[0]["pars"][2:5]

    T_gal = xx_gal + yy_gal

    rho4gal = res_gal[0]["pars"][6]
    flux_gal = res_gal[0]["pars"][5] / res_gal[0]["wnorm"] / obs.jacobian.area

    # Get resi
    resi_img = get_resi_img(
        obs, xx_gal, yy_gal, xy_gal, flux_gal, xx_psf, yy_psf, xy_psf, flux_psf
    )
    resi_obs = obs.copy()
    resi_obs.set_image(resi_img, update_pixels=True)

    guess_resi = np.array(
        [
            res_gal[0]["pars"][0],
            res_gal[0]["pars"][1],
            xx_gal,
            xy_gal,
            yy_gal,
            flux_gal,
        ]
    )
    res_resi = fitter._get_am_result()
    find_ellipmom2(resi_obs.pixels, guess_resi, res_resi, fitter.conf)
    xx_gal, xy_gal, yy_gal = res_resi[0]["pars"][2:5]

    T_gal = xx_gal + yy_gal
    e1_gal = (xx_gal - yy_gal) / T_gal
    e2_gal = (2 * xy_gal) / T_gal
    rho4gal = res_resi[0]["pars"][6]
    flux_gal = res_resi[0]["pars"][5] / res_resi[0]["wnorm"] / obs.jacobian.area

    e1_bj, e2_bj = bj_nullPSF(
        T_psf / T_gal, e1_gal, e2_gal, 0.5 * rho4gal - 1, e1_psf, e2_psf, 0
    )
    xx_final, yy_final, xy_final = get_corrected_mom3(
        e1_bj, e2_bj, T_gal * sqrt(1 - (e1_gal**2 + e2_gal**2))
    )

    T_gal = (xx_gal - xx_psf) + (yy_gal - yy_psf)

    Res = T_gal / T_psf

    return flux_gal, T_gal, Res, xx_final, yy_final, xy_final, 1.0
