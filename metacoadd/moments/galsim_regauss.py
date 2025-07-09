"""
This an implementation of the Galsim re-gauss algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp

This implementation is modified to allowed post-metacalibration PSF correction.
"""

import ngmix
from ngmix.observation import Observation, ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError

from .galsim_admom import (
    DEFAULT_BOUND_CORRECT_WT,
    DEFAULT_MAX_MOMENT_NSIG2,
    DEFAULT_MAXITER,
    DEFAULT_SHIFTMAX,
    DEFAULT_TOL,
    DEFAULT_LAMBDA_ELL,
    DEFAULT_MIN_T_ABS,
)
from .galsim_admom import GAdmomFitter, GAdmomResult, get_result
from .galsim_regauss_nb import (
    _check_exp,
    regauss,
)

from numpy import ndarray

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
        lambda_ell=DEFAULT_LAMBDA_ELL,
        min_T_abs=DEFAULT_MIN_T_ABS,
        rng=None,
    ):
        self.guess_fwhm = guess_fwhm
        self._set_conf(
            maxiter=maxiter,
            shiftmax=shiftmax,
            tol=tol,
            max_moment_nsig2=max_moment_nsig2,
            bound_correct_wt=bound_correct_wt,
            lambda_ell=lambda_ell,
            min_T_abs=min_T_abs,
        )

        self.rng = rng

    def go(self, obs, guess=None):
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
        ares = self._get_am_result(nband)
        scale = mb_obs[0][0].jacobian.scale
        if guess is None:
            guess = self._get_guess(
                scale=scale, nband=nband, guess_fwhm=self.guess_fwhm
            )
        if isinstance(guess, float):
            guess = self._get_guess(scale=scale, nband=nband, guess_fwhm=guess)
        if not isinstance(guess, ndarray):
            raise ValueError(
                "guess must be a float or a numpy.ndarray of shape 5"
            )

        try:
            regauss(
                mb_obs,
                guess,
                ares,
                self._get_am_tmp,
                self.conf,
            )
        except Exception as e:
            ares["flags"] = 2**16

        result = get_result(ares, scale**2, ares["wnorm"][0])

        return GAdmomResult(obs=obs, result=result)
