"""
This is a re-wrote of the adaptive moments from ngmix. They have been slightly
modified to allow error propagation through pseudo-regauss.
"""

import ngmix.flags
import numpy as np
from ngmix.gexceptions import GMixRangeError
from ngmix.gmix import GMixModel
from ngmix.gmix.gmix_nb import GMIX_LOW_DETVAL
from ngmix.observation import Observation
from ngmix.shape import e1e2_to_g1g2
from ngmix.util import get_ratio_error
from numpy import diag

from .ngmix_admom_nb import get_mom_var

DEFAULT_MAXITER = 200
DEFAULT_SHIFTMAX = 5.0  # pixels
DEFAULT_TOL = 1.0e-6
DEFAULT_MAX_MOMENT_NSIG2 = 25
DEFAULT_BOUND_CORRECT_WT = 0.25


class GAdmomResult(dict):
    """
    Represent a fit using adaptive moments, and generate images and mixtures
    for the best fit
    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    result: dict
        the basic fit result, to bad added to this object's keys
    """

    def __init__(self, obs, result):
        self._obs = obs
        self.update(result)

    def get_gmix(self):
        """
        get a gmix representing the best fit, normalized
        """
        if self["flags"] != 0:
            raise RuntimeError("cannot create gmix, fit failed")

        pars = self["pars"].copy()
        pars[5] = 1.0

        T = pars[2] + pars[4]
        e1 = (pars[2] - pars[4]) / T
        e2 = (2.0 * pars[3]) / T

        g1, g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2
        pars[4] = T

        return GMixModel(pars[:-1], "gauss")

    def make_image(self):
        """
        Get an image of the best fit mixture
        Returns
        -------
        image: array
            Image of the model, including the PSF if a psf was sent
        """
        if self["flags"] != 0:
            raise RuntimeError("cannot create image, fit failed")

        obs = self._obs
        jac = obs.jacobian

        gm = self.get_gmix()
        gm.set_flux(obs.image.sum())

        im = gm.make_image(
            obs.image.shape,
            jacobian=jac,
        )
        return im


class GAdmomFitter:
    """
    Measure adaptive moments for the input observation
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

    kind = "admom"

    def __init__(
        self,
        maxiter=DEFAULT_MAXITER,
        shiftmax=DEFAULT_SHIFTMAX,
        tol=DEFAULT_TOL,
        max_moment_nsig2=DEFAULT_MAX_MOMENT_NSIG2,
        bound_correct_wt=DEFAULT_BOUND_CORRECT_WT,
        rng=None,
    ):
        self._set_conf(
            maxiter=maxiter,
            shiftmax=shiftmax,
            tol=tol,
            max_moment_nsig2=max_moment_nsig2,
            bound_correct_wt=bound_correct_wt,
        )

        self.rng = rng

    def go(self, obs, guess_fwhm):
        """
        run the adpative moments
        parameters
        ----------
        obs: Observation
            ngmix.Observation
        guess: ngmix.GMix or a float
            A guess for the fitter.  Can be a full gaussian mixture or a single
            value for T, in which case the rest of the parameters for the
            gaussian are generated.
        """

        from .galsim_admom_nb import find_ellipmom2

        if not isinstance(obs, Observation):
            raise ValueError("input obs must be an Observation")

        guess = self._get_guess(obs=obs, guess_fwhm=guess_fwhm)

        ares = self._get_am_result()

        try:
            find_ellipmom2(
                obs.pixels,
                guess,
                ares,
                self.conf,
            )
        except GMixRangeError:
            ares["flags"] = ngmix.flags.GMIX_RANGE_ERROR

        result = get_result(ares, obs.jacobian.area, ares["wnorm"][0])

        return GAdmomResult(obs=obs, result=result)

    def _get_guess(self, obs, guess_fwhm):
        guess = self._generate_guess(obs=obs, guess_fwhm=guess_fwhm)
        return guess

    def _set_conf(
        self,
        maxiter,
        shiftmax,
        tol,
        max_moment_nsig2,
        bound_correct_wt,
    ):  # noqa
        dt = np.dtype(_Gadmom_conf_dtype, align=True)
        conf = np.zeros(1, dtype=dt)

        conf["maxiter"] = maxiter
        conf["shiftmax"] = shiftmax
        conf["tol"] = tol
        conf["max_moment_nsig2"] = max_moment_nsig2
        conf["bound_correct_wt"] = bound_correct_wt

        self.conf = conf

    def _get_am_result(self):
        dt = np.dtype(_admom_result_dtype, align=True)
        return np.zeros(1, dtype=dt)

    def _get_rng(self):
        if self.rng is None:
            self.rng = np.random.RandomState()

        return self.rng

    def _generate_guess(self, obs, guess_fwhm):  # noqa
        rng = self._get_rng()

        scale = obs.jacobian.get_scale()
        pars = np.zeros(6)
        fwhm2 = guess_fwhm * guess_fwhm
        pars[0 : 0 + 2] = rng.uniform(
            low=-0.5 * scale, high=0.5 * scale, size=2
        )
        pars[2] = fwhm2 * (1.0 + rng.uniform(low=-1e-3, high=1e-3))
        pars[3] = rng.uniform(low=-1e-3, high=1e-3)
        pars[4] = fwhm2 * (1.0 + rng.uniform(low=-1e-3, high=1e-3))
        pars[5] = 1.0

        return pars


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

    res["flux"] = np.nan
    res["flux_mean"] = np.nan
    res["flux_err"] = np.nan
    res["T"] = np.nan
    res["T_err"] = np.nan
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
            # res['flux'] = res['sums'][5] / fnorm
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

    return res


_admom_result_dtype = [
    ("flags", "i4"),
    ("numiter", "i4"),
    ("nimage", "i4"),
    ("npix", "i4"),
    ("wsum", "f8"),
    ("wnorm", "f8"),
    ("sums", "f8", 7),
    ("sums_cov", "f8", (7, 7)),
    ("pars", "f8", 7),
    # temporary
    ("F", "f8", 7),
]

_Gadmom_conf_dtype = [
    ("maxiter", "i4"),
    ("shiftmax", "f8"),
    ("tol", "f8"),
    ("max_moment_nsig2", "f8"),
    ("bound_correct_wt", "f8"),
]
