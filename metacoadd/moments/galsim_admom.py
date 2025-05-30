"""
This an implementation of the Galsim adaptive moments algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp
"""

import ngmix.flags
import numpy as np
from ngmix.gexceptions import GMixRangeError
from ngmix.gmix import GMixModel
from ngmix.gmix.gmix_nb import GMIX_LOW_DETVAL
from ngmix.observation import MultiBandObsList, ObsList, Observation
from ngmix.shape import e1e2_to_g1g2
from ngmix.util import get_ratio_error

from .galsim_admom_nb import (
    compute_effective_flux,
    compute_flux_cross_covs,
    get_mom_var,
)


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
        pars[5] = (
            self["pars"][6] * self["pars"][5] / self._obs.jacobian.scale**2
        )

        T = pars[2] + pars[4]
        e1 = (pars[4] - pars[2]) / T
        e2 = (2.0 * pars[3]) / T

        g1, g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2
        pars[4] = T

        return GMixModel(pars[:6], "gauss")

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
        guess_fwhm,
        psf_deconv=False,
        maxiter=DEFAULT_MAXITER,
        shiftmax=DEFAULT_SHIFTMAX,
        tol=DEFAULT_TOL,
        max_moment_nsig2=DEFAULT_MAX_MOMENT_NSIG2,
        bound_correct_wt=DEFAULT_BOUND_CORRECT_WT,
        rng=None,
    ):
        self.guess_fwhm = guess_fwhm
        self.psf_deconv = psf_deconv
        self._set_conf(
            maxiter=maxiter,
            shiftmax=shiftmax,
            tol=tol,
            max_moment_nsig2=max_moment_nsig2,
            bound_correct_wt=bound_correct_wt,
        )

        self.rng = rng

    def go(self, obs):
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

        if isinstance(obs, MultiBandObsList):
            nband = len(obs)
            mb_obs = obs
        else:
            nband = 1
            if isinstance(obs, ObsList):
                mb_obs = MultiBandObsList()
                mb_obs.append(obs)
            else:
                if isinstance(obs, Observation):
                    nband = 1
                    mb_obs = MultiBandObsList()
                    mb_obs.append(ObsList())
                    mb_obs[0].append(obs)
                else:
                    raise ValueError(
                        "input obs must be a MultiBandObsList or ObsList or Observation"
                    )

        pixels_list = []
        band_tracker = []
        if self.psf_deconv:
            psf_moments = []
            idx = 0
        for i, obslits in enumerate(mb_obs):
            k = 0
            for j, obs_ in enumerate(obslits):
                pixels_list.append(obs_.pixels)
                if self.psf_deconv:
                    psf_obs = obs_.psf
                    if psf_obs.has_gmix():
                        psf_pars = psf_obs.gmix.get_full_pars()
                        psf_moments.append(psf_pars[3:6])
                        idx += 1
                    else:
                        raise ValueError("PSF has no gmix set.")
                k += 1
            band_tracker.append(k)
        band_tracker = np.array(band_tracker)
        if self.psf_deconv:
            psf_moments = np.array(psf_moments)
        else:
            psf_moments = None

        ares = self._get_am_result(nband)
        atmp = self._get_am_tmp(sum(band_tracker))

        scale = mb_obs[0][0].jacobian.scale
        guess = self._get_guess(
            scale=scale, nband=nband, guess_fwhm=self.guess_fwhm
        )

        try:
            find_ellipmom2(
                pixels_list,
                band_tracker,
                guess,
                ares,
                atmp,
                self.conf,
                psf_moments=psf_moments,
            )
        except GMixRangeError:
            # NOTE: Probably need another flag for this
            ares["flags"] = ngmix.flags.GMIX_RANGE_ERROR

        result = get_result(ares, scale**2, ares["wnorm"][0])

        return GAdmomResult(obs=obs, result=result)

    def _get_guess(self, scale, nband, guess_fwhm):
        guess = self._generate_guess(
            scale=scale, nband=nband, guess_fwhm=guess_fwhm
        )
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

    def _get_am_result(self, nband):
        dt = np.dtype(_get_admom_result_dtype(nband), align=True)
        return np.zeros(1, dtype=dt)

    def _get_am_tmp(self, nobs):
        dt = np.dtype(_get_admom_tmp_dtype(nobs), align=True)
        return np.zeros(1, dtype=dt)

    def _get_rng(self):
        if self.rng is None:
            self.rng = np.random.RandomState()

        return self.rng

    def _generate_guess(self, scale, nband, guess_fwhm):  # noqa
        rng = self._get_rng()

        pars = np.zeros(5, dtype=np.float64)
        fwhm2 = guess_fwhm * guess_fwhm
        pars[0 : 0 + 2] = rng.uniform(
            low=-0.5 * scale, high=0.5 * scale, size=2
        )
        pars[2] = fwhm2 * (1.0 + rng.uniform(low=-1e-3, high=1e-3))
        pars[3] = rng.uniform(low=-1e-3, high=1e-3)
        pars[4] = fwhm2 * (1.0 + rng.uniform(low=-1e-3, high=1e-3))
        # pars[5:] = 1.0

        return pars


def get_result(ares, jac_area, wgt_norm):
    """
    Copy the result structure to a dict and calculate a few more things,
    including using Jacobian-based error propagation.
    Now supports multi-band fluxes and covariance.
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
            res[n] = (
                ares[n].reshape((7 + ares["sums"].shape[0] - 7,) * 2).copy()
            )
        else:
            res[n] = ares[n]

    res["flagstr"] = ""
    res["flux_flags"] = 0
    res["flux_flagstr"] = ""
    res["T_flags"] = 0
    res["T_flagstr"] = ""

    res["T"] = np.nan
    res["T_err"] = np.nan
    res["rho4"] = np.nan
    res["rho4_err"] = np.nan
    # res["s2n"] = np.nan
    res["e1"] = np.nan
    res["e2"] = np.nan
    res["e1err"] = np.nan
    res["e2err"] = np.nan
    res["e"] = np.array([np.nan, np.nan])
    res["e_err"] = np.array([np.nan, np.nan])
    res["e_cov"] = np.diag([np.nan, np.nan])

    # Determine number of bands from flux length
    n_bands = res["sums"].shape[0] - 6
    res["n_bands"] = n_bands

    # Allocate multi-band fluxes
    res["flux"] = np.full(n_bands, np.nan)
    res["flux_err"] = np.full(n_bands, np.nan)
    res["flux_cov"] = np.full((n_bands, n_bands), np.nan)
    res["flux_mean"] = np.nan

    if res["flags"] == 0:
        res["T"] = res["pars"][2] + res["pars"][4]
        try:
            flux_eff, flux_eff_var, flux_weights, flux_vars = (
                compute_effective_flux(
                    res["sums"][6:],
                    res["sums_cov"][6:, 6:],
                )
            )
            res["flux_mean"] = flux_eff / (res["wsum"])

            # Also store per-band flux in pars if needed
            res["pars"][6 : 6 + n_bands] = res["sums"][6:] / res["wsum"]
        except np.linalg.LinAlgError:
            # NOTE: need to update to a better flag
            res["T_flags"] |= ngmix.flags.LM_SINGULAR_MATRIX

    if res["flags"] == 0 and res["T"] > GMIX_LOW_DETVAL:
        pnorm = res["pars"][5]
        fnorm = jac_area * wgt_norm * res["wsum"]

        res["flux"][:] = res["pars"][6:] * pnorm / fnorm
        res["flux_err"][:] = np.sqrt(flux_vars) * pnorm / fnorm

        # Store full flux covariance matrix
        for i in range(n_bands):
            for j in range(n_bands):
                i_idx = 6 + i
                j_idx = 6 + j
                res["flux_cov"][i, j] = (
                    res["sums_cov"][i_idx, j_idx] * (pnorm**2) / (fnorm**2)
                )
    else:
        res["flux_flags"] |= res["flags"] | ngmix.flags.NONPOS_SIZE

    if res["flags"] == 0:
        Q11_F_eff_cov, Q22_F_eff_cov = compute_flux_cross_covs(
            flux_weights=flux_weights,
            target_covs=np.array(
                [
                    res["sums_cov"][2, 6:],
                    res["sums_cov"][4, 6:],
                ]
            ),
        )

        if (
            res["sums_cov"][2, 2] > 0
            and res["sums_cov"][4, 4] > 0
            and flux_eff > 0
        ):
            T_var = get_mom_var(
                res["sums"][2],
                res["sums"][4],
                flux_eff,
                res["sums_cov"][2, 2],
                res["sums_cov"][4, 4],
                flux_eff_var,
                res["sums_cov"][2, 4],
                Q11_F_eff_cov,
                Q22_F_eff_cov,
                kind="T",
            )
            res["T_err"] = 4 * np.sqrt(T_var)
        else:
            res["T_flags"] |= ngmix.flags.NONPOS_VAR
    else:
        res["T_flags"] |= res["flags"]

    # Ellipticity and rho4
    if res["flags"] == 0:
        if res["T"] > 0.0:
            res["e1"] = (res["pars"][4] - res["pars"][2]) / res["T"]
            res["e2"] = 2.0 * res["pars"][3] / res["T"]
            res["e"][:] = np.array([res["e1"], res["e2"]])

            e1_var = get_mom_var(
                res["sums"][4],
                res["sums"][2],
                res["sums"][3],
                res["sums_cov"][4, 4],
                res["sums_cov"][2, 2],
                res["sums_cov"][3, 3],
                res["sums_cov"][4, 2],
                res["sums_cov"][4, 3],
                res["sums_cov"][3, 2],
                kind="e1",
            )
            e2_var = get_mom_var(
                res["sums"][4],
                res["sums"][2],
                res["sums"][3],
                res["sums_cov"][4, 4],
                res["sums_cov"][2, 2],
                res["sums_cov"][3, 3],
                res["sums_cov"][4, 2],
                res["sums_cov"][4, 3],
                res["sums_cov"][3, 2],
                kind="e2",
            )
            if np.isfinite(e1_var) and np.isfinite(e2_var):
                res["e1err"] = 2.0 * np.sqrt(e1_var)
                res["e2err"] = 2.0 * np.sqrt(e2_var)
                res["e_err"] = np.array([res["e1err"], res["e2err"]])
                res["e_cov"] = np.diag(res["e_err"] ** 2)
            else:
                res["flags"] |= ngmix.flags.NONPOS_SHAPE_VAR
        else:
            res["flags"] |= ngmix.flags.NONPOS_SIZE

    # handle rho4 for multiband
    if res["flags"] == 0:
        res["rho4"] = res["pars"][5]

        R4_F_eff_cov = compute_flux_cross_covs(
            flux_weights=flux_weights, target_covs=res["sums_cov"][5, 6:]
        )

        if res["sums_cov"][5, 5] > 0 and flux_eff > 0:
            rho4_err = 4 * get_ratio_error(
                res["sums"][5],
                flux_eff,
                res["sums_cov"][5, 5],
                flux_eff_var,
                R4_F_eff_cov,
            )
            res["rho4_err"] = rho4_err
        else:
            res["T_flags"] |= ngmix.flags.NONPOS_VAR

    else:
        res["T_flags"] |= res["flags"]

    if not np.all(np.diag(res["sums_cov"][2:, 2:]) > 0):
        res["flags"] |= ngmix.flags.NONPOS_VAR

    res["flagstr"] = ngmix.flags.get_flags_str(res["flags"])
    res["flux_flagstr"] = ngmix.flags.get_flags_str(res["flux_flags"])
    res["T_flagstr"] = ngmix.flags.get_flags_str(res["T_flags"])
    res["g1"], res["g2"] = e1e2_to_g1g2(res["e1"], res["e2"])
    res["g"] = np.array([res["g1"], res["g2"]])

    return res


def _get_admom_tmp_dtype(nobs):
    _admom_tmp_dtype = [
        ("sums", "f8", (nobs, 7)),
        ("sums_cov", "f8", (nobs, 7, 7)),
        ("flux_jac", "f8", (nobs, 6)),
    ]
    return _admom_tmp_dtype


def _get_admom_result_dtype(nband):
    _admom_result_dtype = [
        ("flags", "i4"),
        ("numiter", "i4"),
        ("nimage", "i4"),
        ("npix", "i4"),
        ("wsum", "f8"),
        ("wnorm", "f8"),
        ("sums", "f8", 6 + nband),
        ("sums_cov", "f8", (6 + nband, 6 + nband)),
        ("pars", "f8", 6 + nband),
        ("s2n", "f8"),
        # temporary
        ("F", "f8", 6 + nband),
    ]
    return _admom_result_dtype


_Gadmom_conf_dtype = [
    ("maxiter", "i4"),
    ("shiftmax", "f8"),
    ("tol", "f8"),
    ("max_moment_nsig2", "f8"),
    ("bound_correct_wt", "f8"),
]
