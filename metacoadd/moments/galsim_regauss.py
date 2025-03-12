import numpy as np
from numpy import diag

import ngmix
from ngmix.observation import Observation, ObsList
from ngmix.shape import e1e2_to_g1g2
from ngmix.util import get_ratio_error
from ngmix.gmix.gmix_nb import GMIX_LOW_DETVAL

from .galsim_admom import (
    DEFAULT_BOUND_CORRECT_WT,
    DEFAULT_MAX_MOMENT_NSIG2,
    DEFAULT_MAXITER,
    DEFAULT_SHIFTMAX,
    DEFAULT_TOL,
)
from .galsim_admom import GAdmomFitter, GAdmomResult
from .galsim_regauss_nb import (
    _check_exp,
    find_ellipmom2,
    regauss,
)
from .ngmix_admom_nb import get_mom_var

DEFAULT_SAFE_CHECK = 0.99


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


class ReGaussFitter(GAdmomFitter):
    """
    Measure re-gauss moments for the input observation
    parameters
    ----------
    maxiter: integer, optional
        Maximum number of iterations, default 200
    etol: float, optional
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T <x^2> + <y^2> to determine
        convergence, default 1.0e-3
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to
        the initial guess.  Default 5.0 (5 pixels if the jacobian
        scale is 1)
    cenonly: bool, optional
        If set to True, only vary the center
    rng: np.random.RandomState
        Random state for creating full gaussian guesses based
        on a T guess
    """

    kind = "regauss"

    def __init__(
        self,
        guess_fwhm,
        maxiter=DEFAULT_MAXITER,
        shiftmax=DEFAULT_SHIFTMAX,
        tol=DEFAULT_TOL,
        max_moment_nsig2=DEFAULT_MAX_MOMENT_NSIG2,
        bound_correct_wt=DEFAULT_BOUND_CORRECT_WT,
        do_safe_check=True,
        safe_check=DEFAULT_SAFE_CHECK,
        rng=None,
    ):
        self.guess_fwhm = guess_fwhm
        self._set_conf(
            maxiter=maxiter,
            shiftmax=shiftmax,
            tol=tol,
            max_moment_nsig2=max_moment_nsig2,
            bound_correct_wt=bound_correct_wt,
            do_safe_check=do_safe_check,
            safe_check=safe_check,
        )

        self.rng = rng

    def go(self, obs, mcal_key=None):
        """
        run re-gauss
        parameters
        ----------
        obs: Observation or ObsList
            ngmix.Observation
        guess: ngmix.GMix or a float
            A guess for the fitter.  Can be a full gaussian mixture or a single
            value for T, in which case the rest of the parameters for the
            gaussian are generated.
        """

        seed = 42

        all_res = []
        if isinstance(obs, Observation):
            res_tmp = self._go_obs(
                obs,
                self.guess_fwhm,
                seed,
                mcal_key,
            )
            jac_area = obs.jacobian.area
            all_res.append(res_tmp)
        elif isinstance(obs, ObsList):
            jac_area = 0
            k = 0
            for obs_ in obs:
                res_tmp = self._go_obs(
                    obs_,
                    self.guess_fwhm,
                    seed,
                    mcal_key,
                )
                all_res.append(res_tmp)
                if res_tmp[0]["flags"] == 0:
                    k += 1
                    jac_area += obs_.jacobian.area
            jac_area /= k
        else:
            raise ValueError("input obs must be an Observation or ObsList")

        res_compile = self.compile_results(all_res)
        result = get_result(res_compile, jac_area, res_compile["wnorm"][0])

        return GAdmomResult(obs=obs, result=result)

    def _go_obs(self, obs, guess_fwhm, seed, mcal_key=None):
        rng = np.random.RandomState(seed)
        fitter = GAdmomFitter(guess_fwhm=guess_fwhm, rng=rng)

        # ares_tmp = self._get_rg_result()
        ares_tmp = self._get_am_result()

        # First, fit PSF
        psf_res = self._get_am_result()
        guess = self._generate_guess(obs.psf, guess_fwhm)
        find_ellipmom2(obs.psf.pixels, guess, psf_res, self.conf)

        # check if exposure is good
        if self.conf["do_safe_check"]:
            w_sum = check_exp(obs, psf_res[0])
        if w_sum < self.conf["safe_check"]:
            # NOTE: Ngmix flags go up to 2**15
            ares_tmp["flags"] = 2**16
        # Now measure gal
        if (ares_tmp[0]["flags"] == 0) & (psf_res[0]["flags"] == 0):
            psf_real = None
            psf_deconv = None
            psf_real_res = None
            if "psf_real" in obs.meta.keys():
                psf_real = obs.meta["psf_real"][mcal_key]
                psf_deconv = obs.meta["psf_deconv"]

                psf_real_res_ = self._get_am_result()
                guess_real = self._generate_guess(psf_real, guess_fwhm)
                find_ellipmom2(
                    obs.psf.pixels, guess_real, psf_real_res_, self.conf
                )
                psf_real_res = psf_real_res_[0]
            regauss(
                obs,
                psf_res[0],
                ares_tmp,
                fitter=fitter,
                guess_fwhm=guess_fwhm,
                psf_real_res=psf_real_res,
                psf_real=psf_real,
                psf_deconv=psf_deconv,
            )
        return ares_tmp, psf_res

    def compile_results(self, all_res):
        rg_res = self._get_rg_result()
        k = 0
        wsum2 = 0
        for res_, res_psf_ in all_res:
            res = res_[0]
            res_psf = res_psf_[0]
            if res["flags"] != 0:
                continue
            rg_res[0]["flags"] |= res["flags"]
            rg_res[0]["wsum"] += res["wsum"]
            rg_res[0]["wnorm"] += res["wnorm"] * res["wsum"]
            rg_res[0]["sums"] += res["sums"] * res["wsum"]
            rg_res[0]["sums_cov"] += res["sums_cov"] * res["wsum"] ** 2
            rg_res[0]["pars"] += res["pars"] * res["wsum"]
            rg_res[0]["pars_psf"] += res_psf["pars"] * res["wsum"]
            wsum2 += res["wsum"] ** 2
            k += 1
        # rg_res[0]["wsum"] /= k
        rg_res[0]["wnorm"] /= rg_res[0]["wsum"]
        rg_res[0]["sums"] /= rg_res[0]["wsum"]
        rg_res[0]["sums_cov"] /= wsum2
        rg_res[0]["pars"] /= rg_res[0]["wsum"]
        rg_res[0]["pars_psf"] /= rg_res[0]["wsum"]

        rg_res[0]["nimage"] = k

        return rg_res

    def _set_conf(
        self,
        maxiter,
        shiftmax,
        tol,
        max_moment_nsig2,
        bound_correct_wt,
        do_safe_check,
        safe_check,
    ):  # noqa
        dt = np.dtype(_regauss_conf_dtype, align=True)
        conf = np.zeros(1, dtype=dt)

        conf["maxiter"] = maxiter
        conf["shiftmax"] = shiftmax
        conf["tol"] = tol
        conf["max_moment_nsig2"] = max_moment_nsig2
        conf["bound_correct_wt"] = bound_correct_wt
        conf["do_safe_check"] = do_safe_check
        conf["safe_check"] = safe_check

        self.conf = conf

    def _get_rg_result(self):
        dt = np.dtype(_regauss_result_dtype, align=True)
        return np.zeros(1, dtype=dt)


def get_result(ares, jac_area, wgt_norm):
    """
    copy the result structure to a dict, and
    calculate a few more things
    """

    if isinstance(ares, np.ndarray):
        ares = ares[0]
        names = ares.dtype.names
    else:
        names = list(ares.keys())

    res = {}
    for n in names:
        if n == "sums":
            res[n] = ares[n].copy()
        elif n == "sums_cov":
            res[n] = ares[n].reshape((7, 7)).copy()
        else:
            res[n] = ares[n]

    res["flagstr"] = ""
    res["flux_flags"] = 0
    res["flux_flagstr"] = ""
    res["T_flags"] = 0
    res["T_flagstr"] = ""

    res["nimage"] = 0

    res["flux"] = np.nan
    res["flux_mean"] = np.nan
    res["flux_err"] = np.nan
    res["Tpsf"] = np.nan
    res["T"] = np.nan
    res["T_err"] = np.nan
    res["Tr"] = np.nan
    res["rho4"] = np.nan
    res["rho4_err"] = np.nan
    res["s2n"] = np.nan
    res["e1"] = np.nan
    res["e2"] = np.nan
    res["e1err"] = np.nan
    res["e2err"] = np.nan
    res["e"] = np.array([np.nan, np.nan])
    res["e_err"] = np.array([np.nan, np.nan])
    res["e_cov"] = np.diag([np.nan, np.nan])

    # set things we always set if flags are ok
    if res["flags"] == 0:
        res["Tpsf"] = res["pars_psf"][2] + res["pars_psf"][4]
        res["T"] = res["pars"][2] + res["pars"][4]
        flux_sum = res["sums"][5]
        res["flux_mean"] = flux_sum / res["wsum"]
        res["pars"][5] = res["flux_mean"]

    # handle flux-only flags
    if res["flags"] == 0:
        if res["T"] > GMIX_LOW_DETVAL:
            # this is a fun set of factors
            # jacobian area is because ngmix works in flux units
            # the wgt_norm and wsum compute the weighted flux and normalize
            # the weight kernel to peak at 1
            fnorm = jac_area * wgt_norm * res["wsum"]
            res["flux"] = res["pars"][5] * res["pars"][6] / fnorm

            if res["sums_cov"][5, 5] > 0:
                res["flux_err"] = (
                    res["pars"][5]
                    * np.sqrt(
                        res["sums_cov"][5, 5] / res["sums"][5] ** 2
                        + res["sums_cov"][6, 6] / res["sums"][6] ** 2
                        + 2
                        * res["sums_cov"][6, 5]
                        / res["sums"][6]
                        / res["sums"][5]
                    )
                    / fnorm
                )
                res["s2n"] = res["flux"] / res["flux_err"]
            else:
                res["flux_flags"] |= ngmix.flags.NONPOS_VAR
        else:
            res["flux_flags"] |= ngmix.flags.NONPOS_SIZE
    else:
        res["flux_flags"] |= res["flags"]

    # handle flux+T only
    if res["flags"] == 0:
        if (
            res["sums_cov"][2, 2] > 0
            and res["sums_cov"][4, 4] > 0
            and res["sums_cov"][5, 5] > 0
        ):
            if res["sums"][5] > 0:
                res["T_err"] = 4 * np.sqrt(
                    get_mom_var(
                        res["sums"][2],
                        res["sums"][4],
                        res["sums"][5],
                        res["sums_cov"][2, 2],
                        res["sums_cov"][4, 4],
                        res["sums_cov"][5, 5],
                        res["sums_cov"][2, 4],
                        res["sums_cov"][2, 5],
                        res["sums_cov"][5, 4],
                        kind="T",
                    )
                )
            else:
                # flux <= 0.0
                res["T_flags"] |= ngmix.flags.NONPOS_FLUX
        else:
            res["T_flags"] |= ngmix.flags.NONPOS_VAR
    else:
        res["T_flags"] |= res["flags"]

    # handle rho4
    if res["flags"] == 0:
        res["rho4"] = res["pars"][6]
        if res["sums_cov"][6, 6] > 0 and res["sums_cov"][5, 5]:
            res["rho4_err"] = 4 * get_ratio_error(
                res["sums"][6],
                res["sums"][5],
                res["sums_cov"][6, 6],
                res["sums_cov"][5, 5],
                res["sums_cov"][6, 5],
            )

    # now handle full flags
    if not np.all(np.diagonal(res["sums_cov"][2:, 2:]) > 0):
        res["flags"] |= ngmix.flags.NONPOS_VAR

    if res["flags"] == 0:
        if res["flux"] > 0:
            if res["T"] > 0.0:
                res["e1"] = (res["pars"][2] - res["pars"][4]) / res["T"]
                res["e2"] = 2.0 * res["pars"][3] / res["T"]
                res["e"][:] = np.array([res["e1"], res["e2"]])

                res["e1err"] = 2.0 * np.sqrt(
                    get_mom_var(
                        res["sums"][2],
                        res["sums"][4],
                        res["sums"][3],
                        res["sums_cov"][2, 2],
                        res["sums_cov"][4, 4],
                        res["sums_cov"][3, 3],
                        res["sums_cov"][2, 4],
                        res["sums_cov"][2, 3],
                        res["sums_cov"][3, 4],
                        kind="e1",
                    )
                )
                res["e2err"] = 2.0 * np.sqrt(
                    get_mom_var(
                        res["sums"][2],
                        res["sums"][4],
                        res["sums"][3],
                        res["sums_cov"][2, 2],
                        res["sums_cov"][4, 4],
                        res["sums_cov"][3, 3],
                        res["sums_cov"][2, 4],
                        res["sums_cov"][2, 3],
                        res["sums_cov"][3, 4],
                        kind="e2",
                    )
                )

                if not np.isfinite(res["e1err"]) or not np.isfinite(
                    res["e2err"]
                ):
                    res["e1err"] = np.nan
                    res["e2err"] = np.nan
                    res["e_err"] = np.array([np.nan, np.nan])
                    res["e_cov"] = diag([np.nan, np.nan])
                    res["flags"] |= ngmix.flags.NONPOS_SHAPE_VAR
                else:
                    res["e_cov"] = diag([res["e1err"] ** 2, res["e2err"] ** 2])
                    res["e_err"] = np.array([res["e1err"], res["e2err"]])

            else:
                res["flags"] |= ngmix.flags.NONPOS_SIZE

        else:
            res["flags"] |= ngmix.flags.NONPOS_FLUX

    res["flagstr"] = ngmix.flags.get_flags_str(res["flags"])
    res["flux_flagstr"] = ngmix.flags.get_flags_str(res["flux_flags"])
    res["T_flagstr"] = ngmix.flags.get_flags_str(res["T_flags"])
    res["g1"], res["g2"] = e1e2_to_g1g2(res["e1"], res["e2"])
    res["g"] = np.array([res["g1"], res["g2"]])
    res["T"] -= res["Tpsf"]
    res["Tr"] = ngmix.moments.get_Tround(res["T"], res["g1"], res["g2"])
    res["nimage"] = ares["nimage"]

    return res


_regauss_result_dtype = [
    ("flags", "i4"),
    ("nimage", "i4"),
    ("wsum", "f8"),
    ("wnorm", "f8"),
    ("sums", "f8", 7),
    ("sums_cov", "f8", (7, 7)),
    ("pars", "f8", 7),
    ("pars_psf", "f8", 7),
]

_regauss_conf_dtype = [
    ("do_safe_check", "?"),
    ("safe_check", "f8"),
    ("maxiter", "i4"),
    ("shiftmax", "f8"),
    ("tol", "f8"),
    ("max_moment_nsig2", "f8"),
    ("bound_correct_wt", "f8"),
]
