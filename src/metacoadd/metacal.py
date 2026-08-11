import numpy as np

import galsim

from ngmix import Observation, ObsList, MultiBandObsList
from ngmix.shape import Shape
from ngmix.metacal.convenience import (
    _replace_image_with_noise,
    _rotate_obs_image_square,
)
from ngmix.metacal.metacal import _check_shape, _do_dilate
from ngmix.metacal.metacal import MetacalFitGaussPSF as MetacalFitGaussPSF_

from ngmix.gexceptions import GMixRangeError


DEFAULT_CONFIG = {
    "step": 0.01,
    "has_pixel": True,
}


class MetacalHandler:
    """Class to handle metacalibration of observations. This class is used to
    make sheared images for use in metacalibration.

    Parameters
    ----------
    rng : np.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    mcal_class : str
        The metacal class to use.  One of 'gauss_psf' or 'fix_gauss_psf'
    fixnoise : bool
        If True, add a noise field to the metacalibrated images to keep the
        noise properties the same as the original image.  If False, the noise
        properties will be different for the metacalibrated images.  This is
        not recommended, as it can lead to biases in the shear measurement.
    use_noise_image : bool
        If True, use the noise image in the observation to add noise to the
        metacalibrated images.  If False, the noise properties will be
        different for the metacalibrated images.  This is not recommended, as
        it can lead to biases in the shear measurement.
    mcal_config : dict
        Additional configuration for the metacal class.See the metacal class
        for details.  This can be used to set the step size, whether to use
        pixel convolution, etc.

    """

    def __init__(
        self,
        rng,
        mcal_class="gauss_psf",
        fixnoise=True,
        use_noise_image=True,
        mcal_config=None,
    ):
        self.rng = rng
        self.fixnoise = fixnoise
        self.use_noise_image = use_noise_image
        self.mcal_config = DEFAULT_CONFIG.copy()
        if mcal_config is not None:
            self.mcal_config.update(mcal_config)

        self._stepk = None
        self._maxk = None

        self._set_mcal_handler(mcal_class)

        self._stepk = None
        self._maxk = None

    def get_all(self, obs, mcal_types):
        """Get all metacalibrated observations for the input observation.

        Parameters
        ----------
        obs : ngmix.Observation, ngmix.ObsList, or ngmix.MultiBandObsList
            The observation to metacalibrate.  If an ObsList or
            MultiBandObsList is provided, the metacalibration will be applied
            to each observation in the list.
        mcal_types : list of str
            The types of metacalibration to apply. In ["noshear", "1p", "1m",
            "2p", "2m"].

        Returns
        -------
        mcal_obs : dict of ngmix.Observation
            A dictionary of metacalibrated observations, with keys
            corresponding to the input mcal_types.

        """
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

        elif isinstance(obs, ObsList):
            mcal_obs = {mcal_type: ObsList() for mcal_type in mcal_types}
            for nobs in obs:
                mcal_obs_ = self.get_all(nobs, mcal_types)
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
                    mcal_obs_ = self.get_all(nobs, mcal_types)
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
        mcal_obs = mcal_maker.get_obs_galshear(mcal_type)

        return mcal_obs

    def _set_mcal_handler(self, mcal_class):
        if mcal_class == "gauss_psf":
            self.mcal_handler = MetacalFitGaussPSF
        elif mcal_class == "fix_gauss_psf":
            self.mcal_handler = MetacalFixGaussPSF
        else:
            raise ValueError(
                "mcal_class must be one of 'gauss_psf' or 'fix_gauss_psf'"
            )

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


class MetacalFixGaussPSF:
    """Make sheared images for use in metacalibration.

    Parameters
    ----------
    obs : ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    step : float
        The shear step size to use for metacalibration.  This is the amount of
        shear to apply to the image for the metacalibration.  Default is 0.01.
    fwhm_target : float
        The target FWHM for the dilated PSF.  This is the FWHM of the Gaussian
        that will be used to dilate the PSF.  The default is 0.3.
    has_pixel : bool
        If True, the pixel convolution is included in the metacalibration.
        If False, the pixel convolution is not included.  Default is True.
    rng : np.random.RandomState
        Random state for generating noise fields. Not needed if metacal if
        using the noise field in the observations.

    """

    def __init__(
        self,
        obs,
        step,
        fwhm_target=0.3,
        has_pixel=True,
        rng=None,
    ):
        self.obs = obs
        self.step = step
        self.has_pixel = has_pixel
        self.rng = rng

        self.fwhm_target = fwhm_target

        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")

        self._set_interp()

        if self.has_pixel:
            self._set_pixel()

        self._set_psf_data(obs)
        self._psf_cache = {}
        # We cache the reconv PSF so we don't have to make it again for the
        # noise image if we use fixnoise
        self._new_psf_obs = None
        self._stepk = None
        self._maxk = None

    def get_obs_galshear(self, mcal_type):
        """This is the case where we shear the image, for calculating R.

        Parameters
        ----------
        mcal_type : str
            The type of shear to apply.  One of ["noshear", "1p", "1m", "2p",
            "2m"]

        Returns
        -------
        newobs : ngmix.Observation
            The metacalibrated observation with the specified shear applied.

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

        newobs = self._make_obs(sheared_image, newpsf_image)

        # this is the pixel-convolved psf object, used to draw the
        # psf image
        newobs.psf.galsim_obj = newpsf_obj
        return newobs

    def _get_psf_key(self, shear, doshear):
        """Need full g1 and g2 in key to support psf shearing."""
        return f"{doshear}-{shear.g1}-{shear.g2}"

    def get_target_image(self, psf_obj, shear=None):
        """Get the target image, convolved with the specified psf
        and possibly sheared.

        Parameters
        ----------
        psf_obj : A galsim object
            psf object by which to convolve.  An interpolated image,
            or surface brightness profile
        shear : ngmix.Shape, optional
            The shear to apply

        Returns
        -------
        galsim image object

        """
        imconv = self._get_target_gal_obj(psf_obj, shear=shear)

        try:
            newim = imconv.drawImage(
                bounds=self._image_bounds,
                wcs=self.get_wcs(),
                method="no_pixel",  # pixel is already in psf
                dtype=np.float64,
            )
        except RuntimeError as err:  # pragma: no cover
            # argh, galsim uses generic exceptions
            raise GMixRangeError(f"galsim error: '{str(err)}'") from err

        return newim

    def _get_target_gal_obj(self, psf_obj, shear=None):
        shim_nopsf = self.get_sheared_image_nopsf(shear)

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        return imconv

    def get_sheared_image_nopsf(self, shear):
        """Get the image sheared by the reqested amount, pre-psf and pre-pixel.

        Parameters
        ----------
        shear : ngmix.Shape
            The shear to apply

        Returns
        -------
        galsim image object

        """
        _check_shape(shear)
        # this is the interpolated, devonvolved image
        sheared_image = self.image_int_nopsf.shear(g1=shear.g1, g2=shear.g2)
        return sheared_image

    def get_target_psf(self, shear, type):
        """Get image and galsim object for the dilated, possibly sheared, psf.

        Parameters
        ----------
        shear : ngmix.Shape
            The applied shear
        type : str
            Type of psf target.  For type='gal_shear', the psf is just dilated
            to deal with noise amplification.  For type='psf_shear' the psf is
            also sheared for calculating Rpsf

        Returns
        -------
        image, galsim object

        """
        _check_shape(shear)

        doshear = type == "psf_shear"

        key = self._get_psf_key(shear, doshear)
        if len(self._psf_cache) == 0 or "True" in key:
            if key not in self._psf_cache:
                psf_grown = galsim.Gaussian(fwhm=self.fwhm_target)
                # Here if we add the back the pixel response we get the
                # wrong shear. I don't understand why
                #     psf_grown = galsim.Convolve(psf_grown, self.pixel)

                try:
                    psf_grown_image = psf_grown.drawImage(
                        bounds=self._psf_bounds,
                        method="no_pixel",  # pixel is already in psf
                        wcs=self.get_psf_wcs(),
                        dtype=np.float64,
                    )
                except RuntimeError as err:  # pragma: no cover
                    # argh, galsim uses generic exceptions
                    raise GMixRangeError(f"galsim error: '{str(err)}'") from err

                self._psf_cache[key] = (psf_grown_image, psf_grown)
        else:
            key = list(self._psf_cache)[0]

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _set_psf_data(self, obs):
        """Create galsim objects based on the input observation."""
        psf_image, psf_int = self._get_interp(
            obs.psf.image.copy(),
            self.get_psf_wcs(),
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
        """Create galsim objects based on the input observation."""
        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        image, image_int = self._get_interp(
            obs.image.copy(), self.get_wcs(), stepk=stepk, maxk=maxk
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

    def get_wcs(self):
        """Get a galsim wcs from the input jacobian."""
        return self.obs.jacobian.get_galsim_wcs()

    def get_psf_wcs(self):
        """Get a galsim wcs from the input jacobian."""
        return self.obs.psf.jacobian.get_galsim_wcs()

    def _set_pixel(self):
        """Set the pixel based on the pixel scale, for convolutions.

        Thanks to M. Jarvis for the suggestion to use toWorld
        to get the proper pixel
        """
        wcs = self.get_wcs()
        self.pixel = wcs.toWorld(galsim.Pixel(scale=1))
        self.pixel_inv = galsim.Deconvolve(self.pixel)

    def _set_interp(self):
        """Set the laczos interpolation configuration."""
        self.interp = "lanczos15"

    def _get_interp(self, img, wcs, stepk=0.0, maxk=0.0):
        """Get a galsim interpolated image object for the input image and wcs.

        Parameters
        ----------
        img : np.ndarray
            The image data
        wcs : galsim wcs
            The wcs to use for the interpolated image
        stepk : float
            The stepk to use for the interpolated image.  If 0.0, the default
            stepk will be used.
        maxk : float
            The maxk to use for the interpolated image.  If 0.0, the default
            maxk will be used.

        Returns
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

    def _make_psf_obs(self, psf_im):
        if self._new_psf_obs is None:
            new_psf_obs = self.obs.psf.copy()
            new_psf_obs.image = psf_im.array
            self._new_psf_obs = new_psf_obs
            return new_psf_obs
        else:
            return self._new_psf_obs

    def _make_obs(self, im, psf_im):
        """Make new Observation objects with the new image and psf.

        Parameters
        ----------
        im : galsim.Image
            The new image
        psf_im : galsim.Image
            The new psf

        Returns
        -------
        newobs : ngmix.Observation
            The new observation with the new image and psf

        """
        newobs = self.obs.copy()
        newobs.image = im.array
        newobs.psf = self._make_psf_obs(psf_im)
        return newobs

    def _clear_data(self):
        del self.image_int_nopsf


class MetacalFitGaussPSF(MetacalFixGaussPSF, MetacalFitGaussPSF_):
    """Make sheared images for use in metacalibration, with a best-fitted
    Gaussian PSF model.

    Parameters
    ----------
    obs : ngmix.Observation
        The observation must have a psf observation set, holding the psf image
    step : float
        The shear step size to use for metacalibration.  This is the amount of
        shear to apply to the image for the metacalibration.  Default is 0.01.
    has_pixel : bool
        If True, the pixel convolution is included in the metacalibration.
        If False, the pixel convolution is not included.  Default is True.
    rng : np.random.RandomState
        Random state for generating noise fields. Not needed if metacal if
        using the noise field in the observations.

    """

    def __init__(
        self,
        obs,
        step,
        has_pixel=True,
        rng=None,
    ):
        super().__init__(
            obs=obs,
            step=step,
            has_pixel=has_pixel,
            rng=rng,
        )

        self._setup_psf_noise()
        self._do_psf_fit()

    def get_target_psf(self, shear, type):
        """Get image and galsim object for the dilated, possibly sheared, psf.

        Parameters
        ----------
        shear : ngmix.Shape
            The applied shear
        type : str
            Type of psf target.  For type='gal_shear', the psf is just dilated
            to deal with noise amplification.  For type='psf_shear' the psf is
            also sheared for calculating Rpsf

        Returns
        -------
        image, galsim object

        """
        _check_shape(shear)

        doshear = type == "psf_shear"

        key = self._get_psf_key(shear, doshear)
        if len(self._psf_cache) == 0 or "True" in key:
            if key not in self._psf_cache:
                psf_grown = self._get_dilated_psf(shear=shear, doshear=doshear)

                try:
                    # psf_grown_ = galsim.Convolve(psf_grown, self.pixel)
                    psf_grown_image = psf_grown.drawImage(
                        bounds=self._psf_bounds,
                        method="no_pixel",  # pixel is already in psf
                        wcs=self.get_psf_wcs(),
                        dtype=np.float64,
                    )
                except RuntimeError as err:  # pragma: no cover
                    # argh, galsim uses generic exceptions
                    raise GMixRangeError(f"galsim error: '{str(err)}'") from err

                self._psf_cache[key] = (psf_grown_image, psf_grown)
        else:
            key = list(self._psf_cache)[0]

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _get_dilated_psf(self, shear, doshear=False):
        """Dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm.
        """
        assert doshear is False, "no shearing fitgauss psf"

        psf_grown = _do_dilate(self.gauss_psf, shear)

        # we don't convolve by the pixel, its already in there
        return psf_grown

    def _make_psf_obs(self, gsim):
        psf_im = gsim.array.copy()

        if self.psf_noise_image is not None:
            psf_im += self.psf_noise_image

        new_psf_obs = self.obs.psf.copy()
        with new_psf_obs.writeable():
            new_psf_obs.image[:, :] = psf_im
            new_psf_obs.weight[:, :] = self.psf_weight

            # Reset the center on the jacobian.
            # We drew the model psf as the exact center
            cen = (np.array(psf_im.shape) - 1.0) / 2.0
            new_psf_obs.jacobian.set_cen(row=cen[0], col=cen[1])

        return new_psf_obs


def _doadd_single_obs(obs, nobs):
    obs.image_orig = obs.image.copy()
    obs.noise_orig = obs.noise.copy()
    obs.weight_orig = obs.weight.copy()

    # the weight and image can be modified in the context, and update_pixels is
    # automatically called upon exit

    with obs.writeable():
        obs.image += nobs.image
        obs.noise += nobs.image

        wpos = np.where((obs.weight != 0.0) & (nobs.weight != 0.0))
        if wpos[0].size > 0:
            tvar = obs.weight * 0
            # add the variances
            tvar[wpos] = 1.0 / obs.weight[wpos] + 1.0 / nobs.weight[wpos]
            obs.weight[wpos] = 1.0 / tvar[wpos]
