from time import time

import numpy as np

from ngmix.fitting.results import FitModel
from ngmix.fitting.fitters import Fitter
from ngmix.fitting.leastsqbound import run_leastsq
from ngmix.gexceptions import GMixRangeError
from ngmix.defaults import LOWVAL, BIGVAL
from ngmix import Observation, ObsList, MultiBandObsList

from .fourier_fitting_nb import (
    zero_pad_fft,
    chisq_from_rfft2_residual,
    fill_fdiff_from_rfft2,
    make_rfft2_onesided_weights,
)
from ..gmix_fourier.gmix_fourier_nb import (
    gmix_eval_fourier_analytic,
    gmix_eval_fourier_analytic_inplace,
)
from ..utils import atleast_mbobs


class FourierFitter(Fitter):
    def __init__(self, model, prior=None, fit_pars=None, stamp_size=None):
        super().__init__(model=model, prior=prior, fit_pars=fit_pars)
        self._stamp_size = stamp_size

    def go(self, obs, guess):
        """
        Run leastsq and set the result

        Parameters
        ----------
        obs: Observation, ObsList, or MultiBandObsList
            Observation(s) to fit
        guess: array
            Array of initial parameters for the fit

        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting.

        """

        n_ps = self._set_ps_iter(obs)
        all_results = []
        for ps_ind in range(n_ps):
            # print("Guess:", guess[4:])
            # ts = time()
            # fit_model_real = self._make_fit_model_real(
            #     obs=obs,
            #     guess=guess,
            # )
            # result_real = run_leastsq(
            #     fit_model_real.calc_fdiff,
            #     guess=guess,
            #     n_prior_pars=fit_model_real.n_prior_pars,
            #     bounds=fit_model_real.bounds,
            #     **self.fit_pars,
            # )
            # if result_real["flags"] == 0:
            #     guess = result_real["pars"]
            fit_model = self._make_fit_model(
                obs=obs, guess=guess, ps_ind=ps_ind
            )
            result = run_leastsq(
                fit_model.calc_fdiff,
                guess=guess,
                n_prior_pars=fit_model.n_prior_pars,
                bounds=fit_model.bounds,
                **self.fit_pars,
            )
            # print(
            #     "Result:",
            #     result["pars"][4:],
            #     "nfev:",
            #     result["nfev"],
            #     # "nfev tot:",
            #     # result_real["nfev"] + result["nfev"],
            #     "time:",
            #     (time() - ts) * 1000,
            #     "snr:",
            #     np.round(result["pars"][-1] / result["pars_err"][-1], 3)
            #     if result["pars_err"][-1] > 0
            #     else -1.0,
            #     # "snr real:",
            #     # np.round(
            #     #     result_real["pars"][-1] / result_real["pars_err"][-1], 3
            #     # )
            #     # if result_real["pars_err"][-1] > 0
            #     # else -1.0,
            # )
            if result["flags"] != 0:
                all_results = [result]
                break
            all_results.append(result)
            guess = result["pars"]

        if len(all_results) > 1:
            result = self._combine_ps_results(all_results)
        else:
            result = all_results[0]

        fit_model.set_fit_result(result)
        return fit_model

    def _make_fit_model(self, obs, guess, ps_ind=0):
        return FourierFitModel(
            obs=obs,
            model=self.model,
            guess=guess,
            prior=self.prior,
            stamp_size=self._stamp_size,
            ps_ind=ps_ind,
        )

    def _make_fit_model_real(self, obs, guess):
        return FitModel(
            obs=obs,
            model=self.model,
            guess=guess,
            prior=self.prior,
        )

    def _combine_ps_results(self, all_results):

        seed = get_seed_from_par(all_results[0]["pars"][0])
        rng = np.random.RandomState(seed)
        res_ind = rng.randint(0, len(all_results))

        final_res = {}
        final_res["flags"] = 0
        for res in all_results:
            final_res["flags"] |= res["flags"]
        final_res["pars"] = np.mean(
            [res["pars"] for res in all_results], axis=0
        )
        final_res["nfev"] = np.max([res["nfev"] for res in all_results])
        final_res["ier"] = all_results[res_ind]["ier"]
        final_res["errmsg"] = all_results[res_ind]["errmsg"]
        final_res["pars_err"] = all_results[res_ind]["pars_err"]
        final_res["pars_cov"] = all_results[res_ind]["pars_cov"]
        final_res["pars_cov0"] = all_results[res_ind]["pars_cov0"]
        return final_res

    def _set_ps_iter(self, obs):
        n_ps = 1
        if isinstance(obs, MultiBandObsList):
            for obslist in obs:
                for obs_ in obslist:
                    ps = obs_.ps
                    if isinstance(ps, list):
                        n_ps = max(n_ps, len(ps))
        elif isinstance(obs, ObsList):
            for obs_ in obs:
                ps = obs_.ps
                if isinstance(ps, list):
                    n_ps = max(n_ps, len(ps))
        else:
            ps = obs.ps
            if isinstance(ps, list):
                n_ps = max(n_ps, len(ps))

        return n_ps


class FourierFitModel(FitModel):
    def __init__(
        self, obs, model, guess, prior=None, stamp_size=None, ps_ind=0
    ):
        super().__init__(obs=obs, model=model, guess=guess, prior=prior)
        self._set_kim(
            atleast_mbobs(obs),
            stamp_size=stamp_size,
            ps_ind=ps_ind,
        )
        self._set_fdiff_size()

    def _get_k_model(self, gm, obs, kim):
        """Return the Fourier-space model for one observation.

        Uses the analytic path (no FFT) when the observation has no mask
        applied (``obs.has_no_mask`` True or mask is all-ones).  Falls back
        to the FFT path when masking is present, because the mask must also
        be applied to the model in that case.

        Parameters
        ----------
        gm : GMix
            Convolved Gaussian mixture for this observation.
        obs : Observation
            The ngmix Observation (used to check masking and get Jacobian).
        kim : ndarray, complex128
            Pre-computed Fourier data for this observation; its shape gives
            the target rfft2 dimension.
        """
        N = kim.shape[0]
        # use_analytic = not self._obs_has_mask(obs)
        use_analytic = True
        if use_analytic:
            j = obs.jacobian
            return gmix_eval_fourier_analytic(
                gm.get_data(),
                N,
                j.row0,
                j.col0,
                j.dvdrow,
                j.dvdcol,
                j.dudrow,
                j.dudcol,
            )
        else:
            r_model = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
            return zero_pad_fft(r_model, target_dim=N)

    @staticmethod
    def _obs_has_mask(obs):
        """Return True if the observation has a non-trivial weight mask."""
        if obs.has_bmask():
            return True
        if obs.has_weight():
            w = obs.weight
            # any zeroed pixel means a masked pixel
            if np.any(w == 0):
                return True
        return False

    def calc_lnprob(self, pars, more=False):

        try:
            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)

            lnprob = 0.0
            s2n_numer = 0.0
            s2n_denom = 0.0
            npix = 0
            self._fill_gmix_all(pars)
            for band in range(self.nband):
                obs_list = self.obs[band]
                gmix_list = self._gmix_all[band]

                for i, (obs, gm) in enumerate(zip(obs_list, gmix_list)):
                    kim = self._kim[band][i]
                    k_model = self._get_k_model(gm, obs, kim)
                    chi2, numer, denom = chisq_from_rfft2_residual(
                        kim, k_model, self._ps[band][i], self._w
                    )

                    lnprob += -0.5 * chi2
                    if more:
                        s2n_numer += numer
                        s2n_denom += denom
                        npix += obs.pixels.size
            lnprob += ln_priors
        except GMixRangeError:
            lnprob = LOWVAL
            s2n_numer = 0.0
            s2n_denom = BIGVAL
            npix = 0

        if more:
            return {
                "lnprob": lnprob,
                "s2n_numer": s2n_numer,
                "s2n_denom": s2n_denom,
                "npix": npix,
            }
        else:
            return lnprob

    def calc_fdiff(self, pars):
        fdiff = np.zeros(self.fdiff_size)

        try:
            self._fill_gmix_all(pars)
            self._fill_priors(pars=pars, fdiff=fdiff)
            # Use n_prior_pars (not _fill_priors return value) so that
            # Fourier data starts exactly where run_leastsq expects it.
            # _fill_priors returns 5 while n_prior_pars is 6 (g1,g2 share
            # one joint prior entry), causing the DC mode to land at index 5
            # and be excluded from the s² calculation in run_leastsq.
            start = self.n_prior_pars
            for band in range(self.nband):
                obs_list = self.obs[band]
                gmix_list = self._gmix_all[band]

                for i, (obs, gm) in enumerate(zip(obs_list, gmix_list)):
                    kim = self._kim[band][i]
                    ps = self._ps[band][i]
                    k_model = self._get_k_model(gm, obs, kim)
                    fill_fdiff_from_rfft2(
                        kim, k_model, ps, self._w, fdiff, start
                    )
                    imsize = kim.size
                    start += imsize * 2

        except GMixRangeError:
            fdiff[:] = LOWVAL
        return fdiff

    def _set_kim(self, obs_in, stamp_size=None, ps_ind=0):
        """
        Store native-resolution Fourier data and PSD.

        The chi² is evaluated at *stamp_size* resolution (the intended full
        stamp size passed from the caller).  When a stamp is clipped at the
        image boundary, ``obs.image`` may be smaller than ``stamp_size`` in
        one or both dimensions.  ``zero_pad_fft`` symmetrically embeds the
        clipped image into a ``stamp_size x stamp_size`` frame before taking
        the FFT.  The model is built at ``obs.image.shape`` using the
        original Jacobian (no centroid adjustment needed) and embedded by the
        same rule inside ``calc_lnprob`` / ``calc_fdiff``, so data and model
        are always identically placed in Fourier space.

        If ``stamp_size`` is None, the dimension is inferred from the first
        observation's image shape (original behaviour for un-clipped stamps).

        The PSD can be provided at either native (stamp_size, stamp_size//2+1)
        or padded (M, M//2+1) resolution; if padded, it is sub-sampled to
        native resolution.
        """

        if stamp_size is not None:
            native_dim = int(stamp_size)
        else:
            native_dim = int(obs_in[0][0].image.shape[0])

        self._kim = []
        self._ps = []
        self.totpix = 0
        for i, obslist in enumerate(obs_in):
            self._kim.append([])
            self._ps.append([])
            for j, obs in enumerate(obslist):
                # FFT at native resolution via numba-accelerated path
                kim = zero_pad_fft(obs.image, target_dim=native_dim)
                self._kim[i].append(kim)
                self.totpix += kim.size

                ps_ = obs.ps
                if isinstance(ps_, list):
                    ps_ij = ps_[ps_ind]
                else:
                    ps_ij = ps_
                native_shape = (native_dim, native_dim // 2 + 1)

                if ps_ij.shape == native_shape:
                    # PSD already at native resolution
                    self._ps[i].append(ps_ij)

        self._w = make_rfft2_onesided_weights(native_dim)

    def _set_fdiff_size(self):
        # we have 2*totpix, since we use both real and imaginary
        # parts
        self.fdiff_size = self.n_prior_pars + 2 * self.totpix


def get_seed_from_par(val):
    power_ = int(np.log10(abs(val)))
    if power_ < 0:
        power = abs(power_) + 4
    if power_ >= 0:
        power = 4 - power_ - 1
    return int(abs(val * 10**power))
