import numpy as np

from ngmix.fitting.results import FitModel
from ngmix.fitting.fitters import Fitter
from ngmix.gexceptions import GMixRangeError
from ngmix.defaults import LOWVAL, BIGVAL

from .fourier_fitting_nb import (
    zero_pad_fft,
    chisq_from_rfft2_residual,
    fill_fdiff_from_rfft2,
    make_rfft2_onesided_weights,
)
from ..utils import atleast_mbobs


class FourierFitter(Fitter):
    def __init__(self, model, prior=None, fit_pars=None, pad_factor=4):
        super().__init__(model=model, prior=prior, fit_pars=fit_pars)
        self._pad_factor = pad_factor

    def _make_fit_model(self, obs, guess):
        return FourierFitModel(
            obs=obs,
            model=self.model,
            guess=guess,
            prior=self.prior,
            pad_factor=self._pad_factor,
        )


class FourierFitModel(FitModel):
    def __init__(self, obs, model, guess, prior=None, pad_factor=4):
        super().__init__(obs=obs, model=model, guess=guess, prior=prior)
        self._pad_factor = pad_factor
        self._set_kim(
            atleast_mbobs(obs),
            pad_factor=pad_factor,
        )
        self._set_fdiff_size()

    def set_fit_result(self, result):
        # Rescale covariance to use real-pixel DOF instead of padded-Fourier DOF.
        # fdiff has 2x padded pixels (real + imag parts); true DOF should use real pixels.
        if result.get("flags", 1) == 0 and result.get("pars_cov0") is not None:
            fdiff = self.calc_fdiff(result["pars"])
            used_dof = fdiff.size - self.n_prior_pars - self.npars

            # Convert padded real-pixel count back to original real-space pixels.
            real_totpix = self._padded_real_totpix // (self._pad_factor**2)
            true_dof = real_totpix - self.npars

            if used_dof > 0 and true_dof > 0:
                sse = np.sum(fdiff[self.n_prior_pars :] ** 2)
                s_sq_true = sse / true_dof
                pars_cov = result["pars_cov0"] * s_sq_true
                result["pars_cov"] = pars_cov
                result["pars_err"] = np.sqrt(np.diag(pars_cov))

        super().set_fit_result(result)

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
                    r_model = gm.make_image(
                        obs.image.shape, jacobian=obs.jacobian
                    )
                    kim = self._kim[band][i]
                    k_model = zero_pad_fft(r_model, target_dim=kim.shape[0])
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
            start = self._fill_priors(pars=pars, fdiff=fdiff)
            for band in range(self.nband):
                obs_list = self.obs[band]
                gmix_list = self._gmix_all[band]

                for i, (obs, gm) in enumerate(zip(obs_list, gmix_list)):
                    r_model = gm.make_image(
                        obs.image.shape, jacobian=obs.jacobian
                    )
                    kim = self._kim[band][i]
                    ps = self._ps[band][i]
                    k_model = zero_pad_fft(r_model, target_dim=kim.shape[0])
                    fill_fdiff_from_rfft2(
                        kim, k_model, ps, self._w, fdiff, start
                    )
                    imsize = kim.size
                    start += imsize * 2

        except GMixRangeError:
            fdiff[:] = LOWVAL
        return fdiff

    def _set_kim(self, obs_in, pad_factor=4):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """

        target_dim = int(obs_in[0][0].image.shape[0] * pad_factor)

        self._kim = []
        self._ps = []
        self.totpix = 0
        self._padded_real_totpix = 0
        for i, obslist in enumerate(obs_in):
            self._kim.append([])
            self._ps.append([])
            for j, obs in enumerate(obslist):
                kim = zero_pad_fft(obs.image, target_dim=target_dim)
                self._kim[i].append(kim)
                self.totpix += kim.size
                self._padded_real_totpix += target_dim * target_dim

                ps_ij = obs.ps
                if ps_ij.shape != kim.shape:
                    raise ValueError(
                        "PSD shape mismatch for band/exposure "
                        f"({i}, {j}): ps={ps_ij.shape}, kim={kim.shape}. "
                        "PSD and Fourier residuals must have identical shape."
                    )
                self._ps[i].append(ps_ij)

        self._w = make_rfft2_onesided_weights(target_dim)

    def _set_fdiff_size(self):
        # we have 2*totpix, since we use both real and imaginary
        # parts
        self.fdiff_size = self.n_prior_pars + 2 * self.totpix
