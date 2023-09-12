import ngmix
import numpy as np

from .galsim_admom import GAdmomFitter
from .galsim_regauss_nb import _check_exp, find_ellipmom2, regauss


def get_psf_fit(obs, fitter, guess_fwhm=1.2, seed=None):
    # PSF
    res_psf = fitter.go(obs.psf, guess_fwhm)
    xx_psf, xy_psf, yy_psf = res_psf["pars"][2:5] / res_psf["wsum"]
    T_psf = xx_psf + yy_psf

    return xx_psf, yy_psf, xy_psf, T_psf


def check_exp(obs, psf_res, safe_factor=2):
    xx_psf, xy_psf, yy_psf = psf_res["pars"][2:5] / psf_res["wsum"]
    T_psf = xx_psf + yy_psf

    e1_psf = (xx_psf - yy_psf) / T_psf
    e2_psf = 2.0 * xy_psf / T_psf

    g1_psf, g2_psf = ngmix.shape.e1e2_to_g1g2(e1_psf, e2_psf)
    pars = [0, 0, g1_psf, g2_psf, safe_factor * T_psf, 1.0]
    weight = ngmix.GMixModel(pars, "gauss")
    w_data = weight._data
    ngmix.gmix.gmix_nb.gmix_set_norms(w_data)

    w_sum = _check_exp(obs.pixels, w_data)
    return w_sum


def ME_regauss(obslist, guess_fwhm=0.6, seed=1234, safe_check=0.99):
    rng = np.random.RandomState(seed)
    fitter = GAdmomFitter(rng=rng)

    # n_epoch = len(obslist)
    # seeds = rng.randint(0, 2**30, size=n_epoch)

    # First, fit PSF and check if exposures are good
    psf_res_list = []
    bad_check_sum = []
    check_sum = []
    for i, obs in enumerate(obslist):
        # psf_res.append(get_psf_fit(obs, fitter, guess_fwhm, seed=seeds[i]))
        psf_res = fitter._get_am_result()
        guess = fitter._generate_guess(obs.psf, guess_fwhm)
        find_ellipmom2(obs.psf.pixels, guess, psf_res, fitter.conf)
        # psf_res.append(fitter.go(obs.psf, guess_fwhm))
        psf_res_list.append(psf_res[0])
        w_sum = check_exp(obs, psf_res_list[i])
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
            obs,
            psf_res_list[i],
            fitter=fitter,
            guess_fwhm=guess_fwhm,
            do_fit=True,
        )
        xx += xx_tmp * w_sum_tmp
        yy += yy_tmp * w_sum_tmp
        xy += xy_tmp * w_sum_tmp
        flux += flux_tmp * w_sum_tmp
        T += T_tmp * w_sum_tmp
        Res += Res_tmp * w_sum_tmp
        norm += w_sum_tmp
        n_good += 1

    # Now deal with bad exposures
    # if len(bad_check_sum):
    #     print("doing bad")
    # bad_obslist = ngmix.ObsList()
    # for i in bad_check_sum:
    #    obs = obslist[i]
    #    pars = [xx/n_good, yy/n_good, xy/n_good]
    #    flux_tmp, T_tmp, Res_tmp, xx_tmp, yy_tmp, xy_tmp, w_sum_tmp = regauss(
    #        obs,
    #        psf_res[i],
    #        fitter=fitter,
    #        pars=pars,
    #        guess_fwhm=guess_fwhm,
    #        do_fit=False)
    #     print("flux_tmp:", flux_tmp)
    #     print(w_sum_tmp)
    #     xx += xx_tmp*w_sum_tmp
    #     yy += yy_tmp*w_sum_tmp
    #     xy += xy_tmp*w_sum_tmp
    #     flux += flux_tmp*w_sum_tmp
    #     T += T_tmp*w_sum_tmp
    #     Res += Res_tmp*w_sum_tmp
    #     norm += w_sum_tmp

    # n_epoch = n_good
    n_good = norm
    xx /= n_good
    yy /= n_good
    xy /= n_good
    flux /= n_good
    T /= n_good
    Res /= n_good

    g1, g2, _ = ngmix.moments.mom2g(yy, xy, xx)
    T = ngmix.moments.get_Tround(T, g1, g2)

    return g1, g2, T, flux, Res
