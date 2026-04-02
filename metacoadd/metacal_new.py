import gc
from copy import deepcopy

import numpy as np

import galsim

import ngmix
from ngmix import Observation, ObsList, MultiBandObsList
from ngmix.shape import Shape
from ngmix.metacal.convenience import (
    _replace_image_with_noise,
    _rotate_obs_image_square,
    _doadd_single_obs,
)
from ngmix.metacal.metacal import _check_shape, _do_dilate
from ngmix.metacal.metacal import MetacalFitGaussPSF as MetacalFitGaussPSF_
from ngmix.simobs import simulate_obs

# from ngmix.metacal.defaults import (
#     DEFAULT_STEP,
#     METACAL_TYPES,
#     METACAL_MINIMAL_TYPES,
# )
from ngmix.gexceptions import GMixRangeError


DEFAULT_CONFIG = {
    "step": 0.01,
    "has_pixel": False,
}


class MetacalHandler:
    def __init__(
        self,
        obs,
        rng,
        mcal_class="gauss_psf",
        fixnoise=True,
        use_noise_image=True,
        mcal_config={},
    ):
        self.rng = rng
        self.fixnoise = fixnoise
        self.use_noise_image = use_noise_image
        self.mcal_config = DEFAULT_CONFIG.copy()
        self.mcal_config.update(mcal_config)

        self._stepk = None
        self._maxk = None

        self._set_mcal_handler(mcal_class)
        # if self.fixnoise:
        #     self._setup_noise_image(obs)

        self._stepk = None
        self._maxk = None

    def get_all(self, obs, mcal_types):

        mcal_obs = self.get_mcal(obs, mcal_types)
        # if self.fixnoise:
        #     noise_obs = self._get_noise_image(obs)
        #     mcal_noise_obs = self.get_mcal(noise_obs, mcal_types)
        #     for mcal_type in mcal_types:
        #         _rotate_obs_image_square(mcal_noise_obs[mcal_type], k=3)
        #         _doadd_obs(mcal_obs[mcal_type], mcal_noise_obs[mcal_type])
        #     del mcal_noise_obs, noise_obs
        return mcal_obs

    def get_mcal(self, obs, mcal_types):

        if isinstance(obs, Observation):
            mcal_obs = {}
            mcal_maker = self.mcal_handler(
                obs,
                rng=self.rng,
                **self.mcal_config,
            )
            for mcal_type in mcal_types:
                mcal_obs[mcal_type] = self._get_one_shear(
                    obs, mcal_maker, mcal_type
                )
            mcal_maker._clear_data()
            del mcal_maker
            gc.collect()

            if self.fixnoise:
                noise_obs = self._get_noise_image(obs)
                mcal_maker = self.mcal_handler(
                    noise_obs,
                    rng=self.rng,
                    **self.mcal_config,
                )
                for mcal_type in mcal_types:
                    mcal_noise_obs = self._get_one_shear(
                        noise_obs, mcal_maker, mcal_type
                    )
                    _rotate_obs_image_square(mcal_noise_obs, k=3)
                    _doadd_single_obs(mcal_obs[mcal_type], mcal_noise_obs)
                mcal_maker._clear_data()
                del mcal_maker, mcal_noise_obs, noise_obs
                gc.collect()

        elif isinstance(obs, ObsList):
            mcal_obs = {mcal_type: ObsList() for mcal_type in mcal_types}
            for nobs in obs:
                mcal_obs_ = self.get_mcal(nobs, mcal_types)
                for mcal_type in mcal_types:
                    mcal_obs[mcal_type].append(mcal_obs_[mcal_type])

        elif isinstance(obs, MultiBandObsList):
            mcal_obs = {
                mcal_type: MultiBandObsList() for mcal_type in mcal_types
            }
            for band_ind, obslist in enumerate(obs):
                for mcal_type in mcal_types:
                    mcal_obs[mcal_type].append(ObsList())
                for nobs in obslist:
                    mcal_obs_ = self.get_mcal(nobs, mcal_types)
                    for mcal_type in mcal_types:
                        mcal_obs[mcal_type][band_ind].append(
                            mcal_obs_[mcal_type]
                        )

        return mcal_obs

    def _get_one_shear(self, obs, mcal_maker, mcal_type):

        if not hasattr(mcal_maker, "image_int_nopsf"):
            stepk, maxk = mcal_maker._set_data(
                obs,
                stepk=self._stepk if self._stepk is not None else 0.0,
                maxk=self._maxk if self._maxk is not None else 0.0,
            )
        if self._stepk is None and self._maxk is None:
            self._stepk = stepk
            self._maxk = maxk
        mcal_obs = mcal_maker.get_obs_galshear(obs, mcal_type)

        return mcal_obs

    def _set_mcal_handler(self, mcal_class):

        if mcal_class == "gauss_psf":
            self.mcal_handler = MetacalFitGaussPSF
        elif mcal_class == "fix_gauss_psf":
            self.mcal_handler = MetacalFixGaussPSF

    def _get_noise_image(self, obs):

        if self.use_noise_image:
            # self._replace_image_with_noise(obs)
            noise_obs = _replace_image_with_noise(obs)
        else:
            raise NotImplementedError(
                "Only use_noise_image=True is implemented at the moment."
            )
        _rotate_obs_image_square(noise_obs, k=1)
        return noise_obs

    # def _replace_image_with_noise(self, obs):
    #     """
    #     Here we setup the noise observation used for fix noise. It is light
    #     weight observation as we only need the noise, the psf is stored from
    #     the original observation and re-used here.
    #     """

    #     if isinstance(obs, Observation):
    #         self.noise_obs = Observation(
    #             image=np.rot90(obs.noise.copy(), k=1),
    #             jacobian=obs.jacobian.copy(),
    #         )
    #     elif isinstance(obs, ObsList):
    #         self.noise_obs = ObsList()
    #         for nobs in obs:
    #             self.noise_obs.append(
    #                 Observation(
    #                     image=np.rot90(nobs.noise.copy(), k=1),
    #                     jacobian=nobs.jacobian.copy(),
    #                 )
    #             )
    #     else:
    #         self.noise_obs = MultiBandObsList()
    #         for obslist in obs:
    #             noise_obslist = ObsList()
    #             for nobs in obslist:
    #                 noise_obslist.append(
    #                     Observation(
    #                         image=np.rot90(nobs.noise.copy(), k=1),
    #                         jacobian=nobs.jacobian.copy(),
    #                     )
    #                 )
    #             self.noise_obs.append(noise_obslist)


class MetacalFixGaussPSF(object):
    """
    Create manipulated images for use in metacalibration

    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image

    examples
    --------

    mc = MetacalFixGaussPSF(obs)

    # observations used to calculate R

    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )

    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)

    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)

    # observations used to calculate Rpsf
    Rpsf_obs1m = mc.get_obs_psfshear(sh1m)
    Rpsf_obs1p = mc.get_obs_psfshear(sh1p)
    Rpsf_obs2m = mc.get_obs_psfshear(sh2m)
    Rpsf_obs2p = mc.get_obs_psfshear(sh2p)
    """

    def __init__(
        self,
        obs,
        step,
        fwhm_target=0.3,
        has_pixel=True,
        rng=None,
    ):

        self._init_obs(obs)
        self.step = step
        self.has_pixel = has_pixel
        self.rng = rng

        self.fwhm_target = fwhm_target

        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")

        self._set_interp()

        if self.has_pixel:
            self._set_pixel(obs)

        self._set_psf_data(obs)
        self._psf_cache = {}
        # We cache the reconv PSF so we don't have to make it again for the
        # noise image if we use fixnoise
        self._new_psf_obs = None
        self._stepk = None
        self._maxk = None

    # def get_all(self, obs, types, _force_stepk=None, _force_maxk=None):

    #     # Setup science data
    #     stepk, maxk = self._set_data(
    #         obs,
    #         stepk=_force_stepk if _force_stepk is not None else 0.0,
    #         maxk=_force_maxk if _force_maxk is not None else 0.0,
    #     )
    #     if self._stepk is None and self._maxk is None:
    #         self._stepk = stepk
    #         self._maxk = maxk

    #     odict = {}
    #     for mcal_type in types:
    #         odict[mcal_type] = self.get_obs_galshear(obs, mcal_type)

    #     # Clear image
    #     self._clear_data()
    #     return odict

    def get_obs_galshear(self, obs, mcal_type):
        """
        This is the case where we shear the image, for calculating R

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply

        get_unsheared: bool
            Get an observation only convolved by the target psf, not
            sheared
        """

        if mcal_type == "noshear":
            shear = Shape(0.0, 0.0)
        elif mcal_type == "1p":
            shear = Shape(self.step, 0.0)
        elif mcal_type == "1m":
            shear = Shape(-self.step, 0.0)
        elif mcal_type == "2p":
            shear = Shape(0.0, self.step)
        elif mcal_type == "2m":
            shear = Shape(0.0, -self.step)

        type = "gal_shear"

        if mcal_type == "noshear":
            newpsf_image, newpsf_obj = self.get_target_psf(
                # We need to dilate the PSF even for noshear
                Shape(self.step, 0.0),
                type,
            )
        else:
            newpsf_image, newpsf_obj = self.get_target_psf(shear, type)

        sheared_image = self.get_target_image(newpsf_obj, shear=shear)

        newobs = self._make_obs(obs, sheared_image, newpsf_image)

        # this is the pixel-convolved psf object, used to draw the
        # psf image
        newobs.psf.galsim_obj = newpsf_obj
        return newobs

    def get_obs_psfshear(self, obs, shear):
        """
        This is the case where we shear the psf image, for calculating Rpsf

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """
        newpsf_image, newpsf_obj = self.get_target_psf(shear, "psf_shear")
        conv_image = self.get_target_image(newpsf_obj, shear=None)

        newobs = self._make_obs(obs, conv_image, newpsf_image)
        return newobs

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm

        If doshear, also shear it
        """

        psf_grown_nopix = self._do_dilate(
            self.psf_int_nopix, shear, doshear=doshear
        )

        if doshear:
            psf_grown_nopix = psf_grown_nopix.shear(g1=shear.g1, g2=shear.g2)

        if self.has_pixel:
            psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
        else:
            psf_grown = psf_grown_nopix
        return psf_grown

    def _do_dilate(self, psf, shear, doshear=False):
        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            self._psf_cache[key] = _do_dilate(psf, shear)

        return self._psf_cache[key]

    def _get_psf_key(self, shear, doshear):
        """
        need full g1 and g2 in key to support psf shearing
        """
        return "%s-%s-%s" % (doshear, shear.g1, shear.g2)

    def get_target_image(self, psf_obj, shear=None, do_noise=False):
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

        imconv = self._get_target_gal_obj(
            psf_obj, shear=shear, do_noise=do_noise
        )

        try:
            newim = imconv.drawImage(
                bounds=self._image_bounds,
                wcs=self.get_wcs(self.obs),
                method="no_pixel",  # pixel is already in psf
            )
        except RuntimeError as err:
            # argh, galsim uses generic exceptions
            raise GMixRangeError(f"galsim error: '{str(err)}'")

        return newim

    def _get_target_gal_obj(self, psf_obj, shear=None, do_noise=False):

        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear, do_noise=do_noise)
        else:
            if not do_noise:
                shim_nopsf = self.image_int_nopsf
            else:
                shim_nopsf = self.noise_image_int_nopsf

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        return imconv

    def get_sheared_image_nopsf(self, shear, do_noise=False):
        """
        get the image sheared by the reqested amount, pre-psf and pre-pixel

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply

        returns
        -------
        galsim image object
        """
        _check_shape(shear)
        # this is the interpolated, devonvolved image
        if not do_noise:
            sheared_image = self.image_int_nopsf.shear(
                g1=shear.g1, g2=shear.g2
            )
        else:
            sheared_image = self.noise_image_int_nopsf.shear(
                g1=shear.g1, g2=shear.g2
            )
        return sheared_image

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
        if len(self._psf_cache) == 0 or "True" in key:
            if key not in self._psf_cache:
                psf_grown_ = galsim.Gaussian(fwhm=self.fwhm_target)
                if self.has_pixel:
                    psf_grown = galsim.Convolve(psf_grown_, self.pixel)
                else:
                    psf_grown = psf_grown_

                try:
                    psf_grown_image = psf_grown.drawImage(
                        bounds=self._psf_bounds,
                        method="no_pixel",  # pixel is already in psf
                        wcs=self.get_psf_wcs(),
                    )
                except RuntimeError as err:
                    # argh, galsim uses generic exceptions
                    raise GMixRangeError(f"galsim error: '{str(err)}'")

                self._psf_cache[key] = (psf_grown_image, psf_grown)
        else:
            key = list(self._psf_cache)[0]

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _set_psf_data(self, obs):
        """
        create galsim objects based on the input observation
        """

        psf_image, psf_int = self._get_interp(
            obs.psf.image.copy(),
            self.get_psf_wcs(obs),
        )
        self._psf_bounds = psf_image.bounds

        # interpolated psf deconvolved from pixel.  This is what
        # we dilate, shear, etc and reconvolve the image by
        if self.has_pixel:
            self.psf_int_nopix = galsim.Convolve(
                psf_int,
                self.pixel_inv,
            )
        else:
            self.psf_int_nopix = psf_int
        self.psf_int_nopix_inv = galsim.Deconvolve(self.psf_int_nopix)

    def _set_data(self, obs, stepk=0.0, maxk=0.0):
        """
        create galsim objects based on the input observation
        """

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        image, image_int = self._get_interp(
            obs.image.copy(), self.get_wcs(obs), stepk=stepk, maxk=maxk
        )
        self._image_bounds = image.bounds

        # deconvolved galaxy image, psf+pixel removed
        if self.has_pixel:
            image_int_nopix = galsim.Convolve(
                image_int,
                self.pixel_inv,
            )
        else:
            image_int_nopix = image_int
        self.image_int_nopsf = galsim.Convolve(
            image_int_nopix,
            self.psf_int_nopix_inv,
        )

        return image_int.stepk, image_int.maxk

    def get_wcs(self, obs):
        """
        get a galsim wcs from the input jacobian
        """
        return obs.jacobian.get_galsim_wcs()

    def get_psf_wcs(self, obs):
        """
        get a galsim wcs from the input jacobian
        """
        return obs.psf.jacobian.get_galsim_wcs()

    def _set_pixel(self, obs):
        """
        set the pixel based on the pixel scale, for convolutions

        Thanks to M. Jarvis for the suggestion to use toWorld
        to get the proper pixel
        """

        wcs = self.get_wcs(obs)
        self.pixel = wcs.toWorld(galsim.Pixel(scale=1))
        self.pixel_inv = galsim.Deconvolve(self.pixel)

        wcs_psf = self.get_psf_wcs(obs)
        self.pixel_psf = wcs_psf.toWorld(galsim.Pixel(scale=1))
        self.pixel_psf_inv = galsim.Deconvolve(self.pixel_psf)

    def _set_interp(self):
        """
        set the laczos interpolation configuration
        """
        self.interp = "lanczos15"

    def _get_interp(self, img, wcs, stepk=0.0, maxk=0.0):
        """
        get a galsim interpolated image object for the input image and wcs

        parameters
        ----------
        image: numpy array
            The image data
        wcs: galsim wcs
            The wcs to use for the interpolated image
        interp: string
            The interpolation method to use, e.g. 'lanczos15'

        returns
        -------
        galsim image, galsim interpolated image object
        """

        image = galsim.Image(img, wcs=wcs)
        image_int = galsim.InterpolatedImage(
            image,
            x_interpolant=self.interp,
            _force_stepk=stepk,
            _force_maxk=maxk,
        )
        return image, image_int

    def _make_psf_obs(self, obs, psf_im):

        if self._new_psf_obs is None:
            new_psf_obs = obs.psf.copy()
            new_psf_obs.image = psf_im.array
            self._new_psf_obs = new_psf_obs
            return new_psf_obs
        else:
            return self._new_psf_obs

    def _make_obs(self, obs, im, psf_im):
        """
        b
        Make new Observation objects with the new image and psf.

        parameters
        ----------
        im: Galsim Image
        psf_im: Galsim Image

        returns
        -------
        A new Observation
        """

        newobs = obs.copy()
        newobs.image = im.array
        newobs.psf = self._make_psf_obs(obs, psf_im)
        return newobs

    def get_interp_param(self):
        """
        Get the stepk and maxk values for the interpolant.
        """
        return self._stepk, self._maxk

    def _clear_data(self):
        del self.image_int_nopsf

    def _init_obs(self, obs):
        """
        Store the input observation for use in making new observations.
        """
        self.obs = DummyObs()
        self.obs.psf = obs.psf
        self.obs._psf = obs._psf
        self.obs.jacobian = obs.jacobian


class MetacalFitGaussPSF(MetacalFixGaussPSF, MetacalFitGaussPSF_):
    def __init__(
        self,
        obs,
        step,
        has_pixel=True,
        rng=None,
    ):
        """
        Parameters
        ----------
        obs: ngmix Observation
            The observation to use for metacal
        rng: numpy.random.RandomState, optional
            Random state for generating noise fields.  Not needed if metacal if
            using the noise field in the observations
        """
        super().__init__(
            obs=obs,
            step=step,
            has_pixel=has_pixel,
            rng=rng,
        )

        self._setup_psf_noise()
        self._do_psf_fit()

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
        if len(self._psf_cache) == 0 or "True" in key:
            if key not in self._psf_cache:
                psf_grown = self._get_dilated_psf(shear=shear, doshear=doshear)

                try:
                    # psf_grown_ = galsim.Convolve(psf_grown, self.pixel)
                    psf_grown_image = psf_grown.drawImage(
                        bounds=self._psf_bounds,
                        method="no_pixel",  # pixel is already in psf
                        wcs=self.get_psf_wcs(self.obs),
                    )
                except RuntimeError as err:
                    # argh, galsim uses generic exceptions
                    raise GMixRangeError(f"galsim error: '{str(err)}'")

                self._psf_cache[key] = (psf_grown_image, psf_grown)
        else:
            key = list(self._psf_cache)[0]

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm
        """

        assert doshear is False, "no shearing fitgauss psf"

        psf_grown = _do_dilate(self.gauss_psf, shear)

        # we don't convolve by the pixel, its already in there
        return psf_grown


class DummyObs:
    def __init__(self):
        self.psf = None
        self.jacobian = None


def _doadd_obs(obs, nobs):

    if isinstance(obs, Observation):
        _doadd_single_obs(obs, nobs)
    elif isinstance(obs, ObsList):
        for o, n in zip(obs, nobs):
            _doadd_single_obs(o, n)
    elif isinstance(obs, MultiBandObsList):
        for obslist, nlist in zip(obs, nobs):
            for o, n in zip(obslist, nlist):
                _doadd_single_obs(o, n)
