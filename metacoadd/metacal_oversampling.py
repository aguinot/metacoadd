import numpy as np

import galsim

import ngmix
from ngmix.metacal.metacal import (
    _galsim_stuff,
    MetacalDilatePSF,
    MetacalGaussPSF,
    MetacalFitGaussPSF,
    MetacalAnalyticPSF,
    _check_shape,
)
from ngmix.metacal.defaults import DEFAULT_STEP
from ngmix.bootstrap import bootstrap
from ngmix.metacal.convenience import (
    _replace_image_with_noise,
    _rotate_obs_image_square,
    _doadd_single_obs,
    # _make_metacal_mb_obs_list_dict,
    _init_mb_obs_list_dict,
    _init_obs_list_dict,
)
from ngmix.observation import (
    Observation,
    ObsList,
    MultiBandObsList,
)
from ngmix.metacal.metacal import _get_ellip_dilation
from ngmix.gexceptions import GMixRangeError
import logging

from .moments.galsim_admom import GAdmomFitter


logger = logging.getLogger(__name__)


class MetacalFitGaussPSFUnderRes(MetacalFitGaussPSF):
    def get_target_image(self, psf_obj, shear=None):
        """
        get the target image, convolved with the specified psf
        and possibly sheared

        parameters
        ----------
        psf_obj: A galsim object
            psf object by which to convolve.  An interpolated image,
            or surface brightness profile
        shear: ngmix.Shape, optional
            The shear to apply

        returns
        -------
        galsim image object
        """

        imconv = self._get_target_gal_obj(psf_obj, shear=shear)

        ny, nx = self.image.array.shape

        # imconv = galsim.Convolve(imconv, self.pixel)

        try:
            newim = imconv.drawImage(
                nx=nx,
                ny=ny,
                # wcs=self.image.wcs,
                wcs=self.get_wcs(),
                # dtype=np.float64,
                # method="no_pixel",
            )
        except RuntimeError as err:
            # argh, galsim uses generic exceptions
            raise GMixRangeError(f"galsim error: '{str(err)}'")

        return newim

    def _get_target_gal_obj(self, psf_obj, shear=None):
        import galsim

        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear)
        else:
            shim_nopsf = self.image_int_nopsf

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        return imconv

    def get_target_psf(self, shear, type):
        """
        get image and galsim object for the dilated, possibly sheared, psf

        parameters
        ----------
        shear: ngmix.Shape
            The applied shear
        type: string
            Type of psf target.  For type='gal_shear', the psf is just dilated
            to deal with noise amplification.  For type='psf_shear' the psf is
            also sheared for calculating Rpsf

        returns
        -------
        image, galsim object
        """

        _check_shape(shear)

        if type == "psf_shear":
            doshear = True
        else:
            doshear = False

        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            # psf_grown = self._get_dilated_psf(shear, doshear=doshear)
            psf_grown = galsim.Gaussian(fwhm=0.3)

            psf_bounds = self.psf_image.bounds

            try:
                psf_grown_ = galsim.Convolve(psf_grown, self.pixel)
                psf_grown_image = psf_grown_.drawImage(
                    bounds=psf_bounds,
                    # image=psf_grown_image,
                    method="no_pixel",  # pixel is already in psf
                    wcs=self.get_psf_wcs(),
                    # wcs=self.get_wcs(),
                )
            except RuntimeError as err:
                # argh, galsim uses generic exceptions
                raise GMixRangeError(f"galsim error: '{str(err)}'")

            self._psf_cache[key] = (psf_grown_image, psf_grown)

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _set_data(self):
        """
        create galsim objects based on the input observation
        """
        import galsim

        obs = self.obs

        # self.interp = "lanczos50"

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        #
        image, image_int = _galsim_stuff(
            obs.image.copy(),
            self.get_wcs(),
            self.interp,
        )
        self.image = image
        self.image_int = image_int

        psf_image, psf_int = _galsim_stuff(
            obs.psf.image.copy(),
            self.get_psf_wcs(),
            self.interp,
        )
        self.psf_image = psf_image

        # this can be used to deconvolve the psf from the galaxy image
        # psf_int_inv = galsim.Deconvolve(psf_int)

        # interpolated psf deconvolved from pixel.  This is what
        # we dilate, shear, etc and reconvolve the image by
        self.psf_int_nopix = galsim.Convolve(
            psf_int,
            self.pixel_inv,
        )
        psf_int_nopix_inv = galsim.Deconvolve(self.psf_int_nopix)

        # deconvolved galaxy image, psf+pixel removed
        image_int_nopix = galsim.Convolve(
            self.image_int,
            self.pixel_inv,
        )
        self.image_int_nopsf = galsim.Convolve(
            image_int_nopix,
            psf_int_nopix_inv,
        )

        # interpolated psf deconvolved from pixel.  This is what
        # we dilate, shear, etc and reconvolve the image by
        self.psf_int = psf_int

    def _set_pixel(self):
        """
        set the pixel based on the pixel scale, for convolutions

        Thanks to M. Jarvis for the suggestion to use toWorld
        to get the proper pixel
        """
        import galsim

        wcs = self.get_wcs()
        self.pixel = wcs.toWorld(galsim.Pixel(scale=1))
        # self.pixel = galsim.Pixel(scale=0.11)
        self.pixel_inv = galsim.Deconvolve(self.pixel)

        wcs_psf = self.get_psf_wcs()
        self.pixel_psf = wcs_psf.toWorld(galsim.Pixel(scale=1))
        self.pixel_psf_inv = galsim.Deconvolve(self.pixel_psf)

    # def _get_dilated_psf(self, shear, doshear=False):
    #     """
    #     dilate the psf by the input shear and reconvolve by the pixel.  See
    #     _do_dilate for the algorithm

    #     If doshear, also shear it
    #     """

    #     psf_grown_nopix = self._do_dilate(self.psf_int_nopix, shear, doshear=doshear)

    #     if doshear:
    #         psf_grown_nopix = psf_grown_nopix.shear(g1=shear.g1, g2=shear.g2)

    #     # psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
    #     return psf_grown_nopix

    # def _do_dilate(self, psf, shear, doshear=False):
    #     key = self._get_psf_key(shear, doshear)
    #     if key not in self._psf_cache:
    #         self._psf_cache[key] = _do_dilate(psf, shear)

    #     return self._psf_cache[key]

    def _do_psf_fit(self):
        """
        do the gaussian fit.

        try the following in order
            - adaptive moments
            - maximim likelihood
            - see if there is already a gmix object


        if the above all fail, rase BootPSFFailure
        """
        import galsim

        # from ngmix.admom import AdmomFitter
        # from ngmix.guessers import GMixPSFGuesser, SimplePSFGuesser
        from ngmix.runners import run_psf_fitter
        # from ngmix.fitting import Fitter

        psfobs = self.obs.psf

        ntry = 4
        # guesser = GMixPSFGuesser(rng=self.rng, ngauss=1)
        guesser = None

        # try adaptive moments first
        # fitter = AdmomFitter(rng=self.rng)
        fitter = GAdmomFitter(rng=self.rng, guess_fwhm=0.1)

        # res = run_psf_fitter(
        #     obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry
        # )

        # e1, e2 = res["e"]
        # T = res["T"]

        # if res['flags'] == 0:
        #     e1, e2 = res['e']
        #     T = res['T']
        # else:
        #     # try maximum likelihood

        #     lm_pars = {
        #         'maxfev': 2000,
        #         'ftol': 1.0e-05,
        #         'xtol': 1.0e-05,
        #     }

        #     fitter = Fitter(model='gauss', fit_pars=lm_pars)
        #     guesser = SimplePSFGuesser(rng=self.rng)

        #     res = run_psf_fitter(
        #         obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry,
        #         set_result=False,
        #     )

        #     if res['flags'] == 0:
        #         psf_gmix = res.get_gmix()
        #     else:

        #         # see if there was already a gmix that we might use instead
        #         if psfobs.has_gmix() and len(psfobs.gmix) == 1:
        #             psf_gmix = psfobs.gmix.copy()
        #         else:
        #             # ok, just raise and exception
        #             raise BootPSFFailure('failed to fit psf '
        #                                  'for MetacalFitGaussPSF')
        #     try:
        #         e1, e2, T = psf_gmix.get_e1e2T()
        #     except GMixRangeError as err:
        #         logger.info('%s', err)
        #         raise BootPSFFailure(
        #             'could not get e1,e2 from psf fit for MetacalFitGaussPSF'
        #         )

        # dilation = _get_ellip_dilation(e1, e2, T)
        # T_dilated = T * dilation
        # sigma = np.sqrt(T_dilated / 2.0)

        # self.gauss_psf = galsim.Gaussian(
        #     sigma=sigma,
        #     flux=self.psf_flux,
        # )


# def _do_dilate(obj, shear):
#     """
#     Dilate the input Galsim image object according to
#     the input shear

#     dilation = 1.0 + 2.0*|g|

#     parameters
#     ----------
#     obj: Galsim Image or object
#         The object to dilate
#     shear: ngmix.Shape
#         The shape to use for dilation
#     """
#     g = np.sqrt(shear.g1**2 + shear.g2**2)
#     dilation = 1.0 + 2.0*g
#     dilation = 1.15
#     return obj.dilate(dilation)


class MetacalBootstrapper:
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model

    Parameters
    ----------
    runner: fit runner for object
        Must have go(obs=obs) method
    psf_runner: fit runner for psfs
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal
    """

    def __init__(
        self,
        runner,
        psf_runner,
        ignore_failed_psf=True,
        rng=None,
        **metacal_kws,
    ):
        self.runner = runner
        self.psf_runner = psf_runner
        self.ignore_failed_psf = ignore_failed_psf
        self.metacal_kws = metacal_kws
        self.rng = rng

    def go(self, obs):
        """
        Run the runners on the input observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        """
        return metacal_bootstrap(
            obs=obs,
            runner=self.runner,
            psf_runner=self.psf_runner,
            ignore_failed_psf=self.ignore_failed_psf,
            rng=self.rng,
            **self.metacal_kws,
        )

    @property
    def fitter(self):
        """
        get a reference to the fitter
        """
        return self.runner.fitter


def metacal_bootstrap(
    obs,
    runner,
    psf_runner=None,
    ignore_failed_psf=True,
    rng=None,
    **metacal_kws,
):
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        Must have go(obs=obs) method
    psf_runner: ngmix PSFRunner, optional
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal

    Returns
    -------
    resdict, obsdict
        resdict is keyed by the metacal types (e.g. '1p') and holds results
        for each

        obsdict is keyed by the metacal types and holds the metacal
        observations

    Side effects
    ------------
    the obs.psf.meta['result'] and the obs.psf.gmix may be set if a psf runner
    is sent and the internal fitter has a get_gmix method.  gmix are only set
    for successful fits
    """

    obsdict = get_all_metacal(obs=obs, rng=rng, **metacal_kws)

    resdict = {}

    for key, tobs in obsdict.items():
        resdict[key] = bootstrap(
            obs=tobs,
            runner=runner,
            psf_runner=psf_runner,
            ignore_failed_psf=ignore_failed_psf,
        )
        # resdict[key] = runner.get_result()

    return resdict, obsdict


def get_all_metacal(
    obs,
    psf="gauss",
    step=DEFAULT_STEP,
    fixnoise=True,
    rng=None,
    use_noise_image=False,
    types=None,
):
    """
    Get all combinations of metacal images in a dict

    parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The values in the dict correspond to these
    psf: string or galsim object, optional
        PSF to use for metacal.  Default 'gauss'.  Note 'fitgauss'
        will usually produce a smaller psf, but it can fail.

            'gauss': reconvolve gaussian that is larger than
                the original and round.
            'fitgauss': fit a gaussian to the PSF and make
                use round, dilated version for reconvolution
            galsim object: any arbitrary galsim object
                Use the exact input object for the reconvolution kernel; this
                psf gets convolved by thye pixel
            'dilate': dilate the origial psf
                just dilate the original psf; the resulting psf is not round,
                so you need to calculate the _psf terms and make an explicit
                correction
    step: float, optional
        The shear step value to use for metacal.  Default 0.01
    fixnoise: bool, optional
        If set to True, add a compensating noise field to cancel the effect of
        the sheared, correlated noise component.  Default True
    rng: np.random.RandomState
        A random number generator; this is required if fixnoise is True and
        use_noise_image is False, or if psf is set to 'fitgauss'.

        If the psf is set to 'gauss', 'fitgauss' or a galsim object, it will be
        used to to add a small amount of noise to the rendered image of the psf
    use_noise_image: bool, optional
        If set to True, use the .noise attribute of the observation
        for fixing the noise when fixnoise=True.
    types: list, optional
        If psf='gauss' or 'fitgauss', then the default set is the minimal
        set ['noshear','1p','1m','2p','2m']

        Otherwise, the default is the full possible set listed in
        ['noshear','1p','1m','2p','2m',
         '1p_psf','1m_psf','2p_psf','2m_psf']

    returns
    -------
    A dictionary with all the relevant metacaled images
        dict keys:
            1p -> ( shear, 0)
            1m -> (-shear, 0)
            2p -> ( 0, shear)
            2m -> ( 0, -shear)
        simular for 1p_psf etc.
    """

    if fixnoise:
        odict = _get_all_metacal_fixnoise(
            obs,
            step=step,
            rng=rng,
            use_noise_image=use_noise_image,
            psf=psf,
            types=types,
        )
    else:
        logger.debug("    not doing fixnoise")
        odict = _get_all_metacal(
            obs,
            step=step,
            rng=rng,
            psf=psf,
            types=types,
        )

    return odict


def _get_all_metacal(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    psf=None,
    types=None,
):
    """
    internal routine

    get all metacal
    """
    if isinstance(obs, Observation):
        if psf == "dilate":
            m = MetacalDilatePSF(obs)
        else:
            if psf == "gauss":
                m = MetacalGaussPSF(obs=obs, rng=rng)
            elif psf == "fitgauss":
                m = MetacalFitGaussPSF(obs=obs, rng=rng)
            elif psf == "fitgauss_UR":
                m = MetacalFitGaussPSFUnderRes(obs=obs, rng=rng)
            else:
                m = MetacalAnalyticPSF(obs=obs, psf=psf, rng=rng)

        odict = m.get_all(step=step, types=types)

    elif isinstance(obs, MultiBandObsList):
        odict = _make_metacal_mb_obs_list_dict(
            mb_obs_list=obs,
            step=step,
            rng=rng,
            psf=psf,
            types=types,
        )
    elif isinstance(obs, ObsList):
        odict = _make_metacal_obs_list_dict(
            obs,
            step,
            rng=rng,
            psf=psf,
            types=types,
        )
    else:
        raise ValueError(
            "obs must be Observation, ObsList, or MultiBandObsList"
        )

    return odict


def _get_all_metacal_fixnoise(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    use_noise_image=False,
    psf=None,
    types=None,
):
    """
    internal routine
    Add a sheared noise field to cancel the correlated noise
    """

    # Using None for the model means we get just noise
    if use_noise_image:
        noise_obs = _replace_image_with_noise(obs)
        logger.debug("    Doing fixnoise with input noise image")
    else:
        noise_obs = ngmix.simobs.simulate_obs(gmix=None, obs=obs, rng=rng)

    # rotate by 90
    _rotate_obs_image_square(noise_obs, k=1)

    obsdict = _get_all_metacal(
        obs,
        step=step,
        rng=rng,
        psf=psf,
        types=types,
    )
    noise_obsdict = _get_all_metacal(
        noise_obs,
        step=step,
        rng=rng,
        psf=psf,
        types=types,
    )

    for type in obsdict:
        imbobs = obsdict[type]
        nmbobs = noise_obsdict[type]

        # rotate back, which is 3 more rotations
        _rotate_obs_image_square(nmbobs, k=3)

        if isinstance(imbobs, Observation):
            _doadd_single_obs(imbobs, nmbobs)

        elif isinstance(imbobs, ObsList):
            for iobs in range(len(imbobs)):
                obs = imbobs[iobs]
                nobs = nmbobs[iobs]

                _doadd_single_obs(obs, nobs)

        elif isinstance(imbobs, MultiBandObsList):
            for imb in range(len(imbobs)):
                iolist = imbobs[imb]
                nolist = nmbobs[imb]

                for iobs in range(len(iolist)):
                    obs = iolist[iobs]
                    nobs = nolist[iobs]

                    _doadd_single_obs(obs, nobs)

    return obsdict


def _make_metacal_mb_obs_list_dict(mb_obs_list, step, rng=None, **kw):
    new_dict = None
    for obs_list in mb_obs_list:
        odict = _make_metacal_obs_list_dict(
            obs_list=obs_list,
            step=step,
            rng=rng,
            **kw,
        )

        if new_dict is None:
            new_dict = _init_mb_obs_list_dict(odict.keys())

        for key in odict:
            new_dict[key].append(odict[key])

    return new_dict


def _make_metacal_obs_list_dict(obs_list, step, rng=None, **kw):
    odict = None
    for obs in obs_list:
        todict = _get_all_metacal(obs, step=step, rng=rng, **kw)

        if odict is None:
            odict = _init_obs_list_dict(todict.keys())

        for key in odict:
            odict[key].append(todict[key])

    return odict
