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
    def __init__(self, model, prior=None, fit_pars=None, stamp_size=None):
        super().__init__(model=model, prior=prior, fit_pars=fit_pars)
        self._stamp_size = stamp_size

    def _make_fit_model(self, obs, guess):
        return FourierFitModel(
            obs=obs,
            model=self.model,
            guess=guess,
            prior=self.prior,
            stamp_size=self._stamp_size,
        )


class FourierFitModel(FitModel):
    def __init__(self, obs, model, guess, prior=None, stamp_size=None):
        super().__init__(obs=obs, model=model, guess=guess, prior=prior)
        self._set_kim(
            atleast_mbobs(obs),
            stamp_size=stamp_size,
        )
        self._set_fdiff_size()

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

    def _set_kim(self, obs_in, stamp_size=None):
        """
        Store native-resolution Fourier data and PSD.

        The chi² is evaluated at *stamp_size* resolution (the intended full
        stamp size passed from the caller).  When a stamp is clipped at the
        image boundary, ``obs.image`` may be smaller than ``stamp_size`` in
        one or both dimensions.  ``zero_pad_fft`` symmetrically embeds the
        clipped image into a ``stamp_size × stamp_size`` frame before taking
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

                ps_ij = obs.ps
                native_shape = (native_dim, native_dim // 2 + 1)

                if ps_ij.shape == native_shape:
                    # PSD already at native resolution
                    self._ps[i].append(ps_ij)

        self._w = make_rfft2_onesided_weights(native_dim)

    def _set_fdiff_size(self):
        # we have 2*totpix, since we use both real and imaginary
        # parts
        self.fdiff_size = self.n_prior_pars + 2 * self.totpix
