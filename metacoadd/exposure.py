import copy
from typing import Optional, Self

import galsim
import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp

from metacoadd.utils import WCSBundle, shift_wcs

DEFAULT_INTERP_CONFIG = {
    "classic": {
        "pad_factor": 4,
        "x_interpolant": "quintic",
        "k_interpolant": "quintic",
        "calculate_maxk": False,
        "calculate_stepk": False,
    },
    "nearest": {
        "pad_factor": 4,
        "x_interpolant": "nearest",
        "k_interpolant": "nearest",
        "calculate_maxk": False,
        "calculate_stepk": False,
    },
}

DEFAULT_RESAMP_CONFIG = {
    "classic": {
        "order": 5,
    },
    "nearest": {
        "order": 0,
    },
}

DEFAULT_GSPARAMS = galsim.GSParams(maximum_fft_size=8192)


class Exposure:
    """Exposure

    Structure to store all the informations for an exposure.

    TODO: Add consistancy check if several images are provided:
        Same size. Other?

    Args:
        image (numpy.ndarray or galsim.Image): Science image.
        header (astropy.io.fits.header.Header): Image header containing all
            the WCS information. Either header or wcs has to be provided, not
            both.
        wcs (galsim.BaseWCS or astropy.wcs.WCS or WCSBundle): wcs corresponding
            to the images. Either header or wcs has to be provided, not both.
        weight (numpy.ndarray or galsim.Image, optional): Weight image.
            Defaults to None.
        flag (numpy.ndarray or galsim.Image, optional): Flag image. Defaults to
            None.
        noise (numpy.ndarray or galsim.Image, optional): Noise image. Defaults
            to None.
        set_meta (bool): If `True` will set the metadata information during the
            initialization. Should always be `True`. This is mainly to be able
            to probagate those information when a resizing is apply through
            `__getitem__`. Defaults to True.
    """

    wcs_bundle: WCSBundle
    image: galsim.Image
    weight: Optional[galsim.Image] = None
    flag: Optional[galsim.Image] = None
    noise: Optional[galsim.Image] = None

    def __init__(
        self,
        image: np.ndarray | galsim.Image,
        wcs: WCS | galsim.BaseWCS | WCSBundle,
        weight=None,
        flag=None,
        noise=None,
        set_meta=True,
    ):
        self._exposure_images = []

        if isinstance(wcs, WCSBundle):
            self.wcs_bundle = wcs
        else:
            self.wcs_bundle = WCSBundle(wcs, image)

        self._init_input_image(image, "image")
        if weight is not None:
            self._init_input_image(weight, "weight")
        if flag is not None:
            self._init_input_image(flag, "flag")
        if noise is not None:
            self._init_input_image(noise, "noise")

        if set_meta:
            self._set_meta()

    def __getitem__(self, bounds: galsim.BoundsI) -> Self:
        """
        Return a new Exposure instance with the corresponding subimages.
        Also handle the WCS.

        Args:
            bounds (galsim.BoundsI): New bounds for the images.

        Returns:
            Exposure: a new Exposure instance.
        """
        if not isinstance(bounds, galsim.BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI.")
        new_exp_dict = {}
        for image_kind in self._exposure_images:
            new_exp_dict[image_kind] = copy.deepcopy(getattr(self, image_kind))[
                bounds
            ]

        # We need to update the WCS to match new origin
        # WARNING: only if the origin changes
        orig_wcs = copy.deepcopy(self.wcs_bundle)
        if self._meta["image_bounds"] != bounds:
            offset_wcs = galsim.PositionI(bounds.xmin, bounds.ymin)
            new_exp_dict["wcs"] = shift_wcs(orig_wcs, offset_wcs)
            for image_kind in self._exposure_images:
                new_exp_dict[image_kind].wcs = new_exp_dict["wcs"].galsim
        else:
            # If same bounds we still run the shift but we set the shift to
            # the fits origin.
            new_exp_dict["wcs"] = shift_wcs(orig_wcs, galsim.PositionI(1, 1))
            for image_kind in self._exposure_images:
                new_exp_dict[image_kind].wcs = new_exp_dict["wcs"].galsim

        new_exposure = Exposure(set_meta=False, **new_exp_dict)
        new_exposure._meta = copy.deepcopy(self._meta)

        return new_exposure

    def _set_galsim_image(self, image):
        """Set GalSim image

        Transform the input image array as a galsim.Image.
        Args:
            image (numpy.ndarray): Image to transform to a galsim.Image.
        Returns:
            galsim.Image: The corresponding galsim.Image.
        """

        galsim_image = galsim.Image(
            image,
            xmin=1,
            ymin=1,
            wcs=self.wcs_bundle.galsim,
            copy=True,
        )

        return galsim_image

    def _init_input_image(self, image, image_kind):
        """Set input image

        Check if the input image is a valid input and add it to Exposure.

        Args:
            image (numpy.ndarray or galsim.Image): Image to setup.
            image_knid (str): Name of the image to set. Must be in ['image',
                'weight', 'flag', 'noise'].
        """

        if isinstance(image, np.ndarray):
            galsim_image = self._set_galsim_image(image)
        elif isinstance(image, galsim.Image):
            if image.wcs is None:
                image.wcs = self.wcs_bundle.galsim
            elif image.wcs != self.wcs_bundle.galsim:
                raise ValueError(
                    "Inconsistant WCS between galsim.Image({image_kind}) and "
                    "provided wcs."
                )
            galsim_image = image
        else:
            raise TypeError(
                "image must be a ngmix.ndarray or a galsim.Image. "
                f"Got {type(image)}."
            )
        self._exposure_images.append(image_kind)
        setattr(self, image_kind, galsim_image)

    def _set_meta(self):
        """
        Set metadata information.
        At moment, save only the image bounds.
        """

        self._meta = {
            "image_bounds": self.image.bounds,
        }


class CoaddImage:
    """CoaddImage

    Structure to store all the informations to build a coadd image.
    This class do not build the coadd but will prepare the data (create
    interpolated images).

    Args:
        explist (metacoadd.ExpList): ExpList object that store all the exposure
            to build the coadd. It can also include images that do not
            contribute to the coadd area and they will be automatically
            ignored.
        world_coadd_center (galsim.celestial.CelestialCoord): Position of the
            coadd center in world coordinates.
        scale (float): Pixel scale of the coadd. In arcsec.
        image_coadd_size (tuple, list or int): Size of the coadd in
            image coordinates. If a `int` is provided, will assume the coadd to
            be square.  Otherwise, has to be a `list` or `tuple` of `int`.
            Either `image_coadd_size` or `world_coadd_size` as to be provided.
        world_coadd_size (tuple, list or galsim.angle.Angle): Size of the coadd
            in world coordinates, in arcmin. If a `galsim.angle.Angle` is
            provided, will assume the coadd to be square. Otherwise, has to be
            a `list` or `tuple` of `galsim.angle.Angle`. Either
            `image_coadd_size` or `world_coadd_size` as to be provided.
        interp_config (dict, optional): Set of parameters for the
            interpolation. If `None` use the default configuration. Defaults to
            None.
        resamp_config (dict, optional): Set of parameters for the resampling.
            If `None` use the default configuration. Defaults to None.
        resize_exposure (bool, optional): Whether to resize the exposures
            before doing the interpolation. It is recommanded to leave this to
            `True` since it will save computing time and memory. We use a
            "relax" parameters to make the resizing slightly larger the the
            coadd size given that this operation happen before the
            interpolation. This avoid to cut a part of the exposure due to
            projection effect later. See `relax_resize`. This is not
            related to the padding for the interpolation. Defaults to True.
        relax_resize (float, optional): Default relax parameters for
            the resizing (see above). Correspond to a percentage of the coadd
            size for both axes. Has to be in ]0, 1] (no good reason to go for
            more than 1 given that distrotion effect are small). This can be
            internally change in case we reach one of the border of the
            exposure. Default to 0.10.
        gsparams ([type], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        explist: list[Exposure],
        world_coadd_center: galsim.celestial.CelestialCoord,
        scale,
        image_coadd_size=None,
        world_coadd_size=None,
        interp_config=None,
        resamp_config=None,
        resize_exposure=True,
        relax_resize=0.10,
        gsparams=None,
    ):
        if all(isinstance(exp, Exposure) for exp in explist):
            self._orig_explist = explist
        else:
            raise TypeError("explist has to be a list of metacoadd.Exposure.")

        if isinstance(world_coadd_center, galsim.celestial.CelestialCoord):
            self.world_coadd_center = world_coadd_center
        else:
            raise TypeError(
                "world_coadd_center has to be a "
                "galsim.celestial.CelestialCoord"
            )

        if image_coadd_size is not None:
            if world_coadd_size is not None:
                raise ValueError(
                    "Either image_coadd_center or world_coadd_center has to "
                    "be provided, not both."
                )
            if isinstance(image_coadd_size, list) or isinstance(
                image_coadd_size, tuple
            ):
                if all(isinstance(n, int) for n in image_coadd_size):
                    self.image_coadd_size = list(image_coadd_size)
                else:
                    raise TypeError(
                        "image_coadd_size has to be a list or tuple of int."
                    )
            elif isinstance(image_coadd_size, int):
                self.image_coadd_size = [image_coadd_size] * 2
            else:
                raise TypeError(
                    "image_coadd_size has to be a list, tuple or int."
                )
        elif world_coadd_size is not None:
            if isinstance(world_coadd_size, list) or isinstance(
                world_coadd_size, tuple
            ):
                if all(
                    isinstance(n, galsim.angle.Angle) for n in world_coadd_size
                ):
                    self._set_image_coadd_size(list(world_coadd_size), scale)
                else:
                    raise TypeError(
                        "image_coadd_size has to be a list or tuple of int."
                    )
            elif isinstance(world_coadd_size, galsim.angle.Angle):
                self._set_image_coadd_size([world_coadd_size] * 2, scale)
            else:
                raise TypeError(
                    "world_coadd_size has to be a list, tuple or "
                    "galsim.angle.Angle."
                )
        else:
            raise ValueError(
                "Either image_coadd_size or world_coadd_size has to be "
                "provided."
            )

        # Set galsim bounds and dervie center for the coadd
        self._set_coadd_bounds()
        self._set_image_coadd_center()

        # Set coadd WCS
        if isinstance(scale, float):
            self._set_coadd_wcs(scale)
        else:
            TypeError("scale has to be a float.")

        # Resize the exposures if requested
        self._relax_resize = None
        if resize_exposure:
            if relax_resize is None:
                raise ValueError(
                    "relax_resize has to be provided to resize exposure."
                )
            if isinstance(relax_resize, float):
                if relax_resize > 0.0 and relax_resize <= 1.0:
                    self.resize_explist(relax_resize)
                    self._relax_resize = relax_resize
                else:
                    raise ValueError("relax_resize has to be in ]0, 1].")
            else:
                raise TypeError("relax_resize has to be a float.")
        else:
            self.resize_explist(0)

        if interp_config is None:
            self.interp_config = DEFAULT_INTERP_CONFIG
        else:
            if isinstance(interp_config, dict):
                self.interp_config = interp_config
            else:
                raise TypeError("interp_config must be a dict.")

        if resamp_config is None:
            self.resamp_config = DEFAULT_RESAMP_CONFIG
        else:
            if isinstance(resamp_config, dict):
                self.resamp_config = resamp_config
            else:
                raise TypeError("resamp_config must be a dict.")

        if gsparams is None:
            self._gsparams = DEFAULT_GSPARAMS
        else:
            if isinstance(interp_config, galsim.GSParams):
                self._gsparams = gsparams
            else:
                raise TypeError("gsparams must be a galsim.GSParams.")

    def _set_image_coadd_size(self, world_coadd_size, scale):
        """Set coadd size

        Set the size of the coadd in pixels from angle.

        Args:
            world_coadd_size (list): List of `galsim.angle.Angle`.
            scale (float): Coadd pixel scale
        """

        from math import ceil

        size_x = ceil((world_coadd_size[0] / galsim.arcmin) / scale)
        size_y = ceil((world_coadd_size[1] / galsim.arcmin) / scale)

        self.image_coadd_size = [size_x, size_y]

    def _set_coadd_bounds(self):
        """
        Create a galsim.Image that describe the coadd. This is just for
        convinience.
        """

        self.coadd_bounds = galsim.BoundsI(
            xmin=1,
            xmax=self.image_coadd_size[0],
            ymin=1,
            ymax=self.image_coadd_size[1],
        )

    def _set_image_coadd_center(self):
        """
        Set coadd center in pixel.
        """

        self.image_coadd_center = self.coadd_bounds.true_center

    def _set_coadd_wcs(self, scale):
        """Set coadd wcs

        Set the coadd WCS as TAN projection with the given pixel scale.

        Args:
            scale (float): Coadd pixel scale.
        """

        # Here we shift the center to match conventions
        affine_transform = galsim.AffineTransform(
            scale,
            0.0,
            0.0,
            scale,
            origin=self.image_coadd_center,  # -galsim.PositionD(1., 1.),
        )

        coadd_wcs_galsim = galsim.TanWCS(
            affine=affine_transform,
            world_origin=self.world_coadd_center,
            units=galsim.arcsec,
        )

        self.coadd_wcs_bundle = WCSBundle(
            coadd_wcs_galsim, np.zeros(self.image_coadd_size)
        )
        self.coadd_pixel_scale = scale

    def resize_explist(self, relax_resize):
        """Resize exposure list

        Args:
            relax_resize (float): Resize relax parameter.
        """
        resized_explist = []
        for exp in self._orig_explist:
            resized_exp = self._resize_exp(exp, relax_resize)
            if resized_exp is not None:
                # Might be weird to set this here, but this function is called
                # automatically and we will need this parameter later.
                # NOTE: Maybe find a better way to do this..
                resized_exp._interp = False
                resized_exp._resamp = False
                resized_explist.append(resized_exp)

        if len(resized_explist) == 0:
            raise ValueError(
                "None of the provided exposure overlap with the coadd area."
            )
        else:
            self.explist = resized_explist

    def _resize_exp(self, exp, relax_resize):
        """Resize exposure

        Args:
            exp (metacoadd.Exposure): Exposure to resize.
            relax_resize (float): Resize relax parameter.
        Returns:
            metacoadd.Exposure or `None`: Return the resized exposure or None
                if the exposure is not in the coadd footprint.
        """

        exp_bounds = exp.image.bounds

        # Here we need to round the position of the coadd but this is just to
        # compute a rough footprint of the coadd on the exposure. We will
        # estimate this latter with a better accuracy.
        # NOTE: This approximation might lead to remove an exposure that was
        #       in the coadd footprint but it would have contribute for a few
        #       pixels (a line or a column maximum).
        try:
            image_coadd_center_on_exp = exp.image.wcs.toImage(
                self.world_coadd_center
            )
        except TypeError:
            world_pos = galsim.PositionD(
                self.world_coadd_center.ra.deg,
                self.world_coadd_center.dec.deg,
            )
            image_coadd_center_on_exp = exp.image.wcs.toImage(world_pos)

        # Make raw bounds
        new_bounds = galsim.BoundsI(
            xmin=1,
            xmax=int(self.coadd_bounds.xmax * (1.0 + relax_resize)),
            ymin=1,
            ymax=int(self.coadd_bounds.ymax * (1.0 + relax_resize)),
        )

        # Shift the bounds to the correct position
        new_bounds = new_bounds.shift(
            image_coadd_center_on_exp.round() - new_bounds.center
        )

        # First check if there is an overlap between the coadd footprint and
        # and the exposure
        overlap = exp_bounds & new_bounds
        if not overlap.isDefined():
            return None
        # Now check if the entire coadd footprint is within the exposure
        if exp_bounds.includes(new_bounds):
            return exp[new_bounds]
        # if not, we cut the coadd footprint at the exposure edges
        else:
            new_bounds = new_bounds & exp.image.bounds
            return exp[new_bounds]

    def get_all_resamp_images(
        self,
        **kwargs,
    ):
        """
        Get all resampeled images.

        Args:
            kwargs: Any additionnal keywords arguments for
                reproject.reproject_interp.
        """

        for exp in self.explist:
            for image_kind in exp._exposure_images:
                # We skip if an image is already resampled
                if "_resamp" in image_kind:
                    continue

                if image_kind == "weight" or image_kind == "flag":
                    resamp_method = "nearest"
                else:
                    resamp_method = "classic"

                # NOTE: Add verbose option
                # print(f"Interpolate {image_kind}...")
                input_reproj = (
                    getattr(exp, image_kind).array,
                    exp.wcs_bundle.astropy,
                )
                resampled, footprint = self._do_resamp(
                    input_reproj,
                    resamp_method,
                    **kwargs,
                )
                gal_img_tmp = galsim.Image(
                    resampled,
                    wcs=copy.deepcopy(self.coadd_wcs_bundle.galsim),
                )
                print(image_kind)
                try:
                    print(gal_img_tmp.FindAdaptiveMom())
                except galsim.errors.GalSimHSMError:
                    print("Mom failed")
                setattr(exp, image_kind + "_resamp", gal_img_tmp)
            exp.wcs_bundle_resamp = copy.deepcopy(self.coadd_wcs_bundle)

            exp._resamp = True

    def _do_resamp(self, input, resamp_method, **kwargs):
        resamp_config = self.resamp_config[resamp_method]
        resamp_config.update(kwargs)

        print(input[1])
        print(self.coadd_wcs_bundle.astropy)

        resamp_img, footprint = reproject_interp(
            input,
            output_projection=self.coadd_wcs_bundle.astropy,
            shape_out=self.image_coadd_size,
            **resamp_config,
        )

        resamp_img[np.isnan(resamp_img)] = 0

        return resamp_img, footprint

    def get_all_interp_images(
        self,
        **kargs,
    ):
        """
        Get all interpolated images.

        Args:
            kwargs: Any additionnal keywords arguments for
                galsim.InterpolatedImage.
        """

        for exp in self.explist:
            # galsim.InterpolatedImage works with local WCS so we force it to
            # take the one at the center of the coadd for better accuracy.
            # This probably do not make a difference but it is more
            # "elegant" to do it :)
            if exp.wcs_bundle.galsim.isLocal():
                wcs = exp.wcs_bundle.galsim
            else:
                wcs = exp.wcs_bundle.galsim.local(
                    world_pos=self.world_coadd_center
                )
            for image_kind in exp._exposure_images:
                if image_kind == "weight" or image_kind == "flag":
                    interp_method = "nearest"
                else:
                    interp_method = "classic"

                # NOTE: Add verbose option
                # print(f"Interpolate {image_kind}...")
                interpolated = self._do_interp(
                    getattr(exp, image_kind),
                    wcs,
                    interp_method,
                    **kargs,
                )
                setattr(exp, image_kind + "_interp", interpolated)
            exp._interp = True

    def _do_interp(self, image, wcs, interp_method, **kwargs):
        """Run interpolation

        Interpolate an image using galsim and return a
        `galsim.interpolatedimage.InterpolatedImage`. This method use the wcs
        set in the initialization by default.

        Args:
            image (numpy.ndarray): Image to interpolate.
            wcs (galsim.BaseWCS): WCS of the Image.
            interp_method (str): Select which interpolation to use. Hard coded.
                Must be in ['classic', 'weight', 'flag']. Will set the
                interpolation to 'nearest' for the weight and flag images.
            kwargs: Any additionnal keywords arguments for
                galsim.InterpolatedImage.

        Returns:
            galsim.interpolatedimage.InterpolatedImage: Interpolated image.
        """

        interp_config = self.interp_config[interp_method]
        interp_config.update(kwargs)

        interp_image = galsim.InterpolatedImage(
            image,
            wcs=wcs,
            gsparams=self._gsparams,
            **interp_config,
        )

        return interp_image

    def setup_coadd(self):
        """
        Setup the coadd image and weight.
        """

        self.image = galsim.Image(
            bounds=self.coadd_bounds,
            wcs=self.coadd_wcs_bundle.galsim,
        )
        # Initially the image is fill with zeros to avoid issues in case the
        # exposures does not fill the entire footprint.
        self.image.fill(0)

        self.noise = galsim.Image(
            bounds=self.coadd_bounds,
            wcs=self.coadd_wcs_bundle.galsim,
        )
        # Initially the noise is fill with zeros to avoid issues in case the
        # exposures does not fill the entire footprint.
        self.noise.fill(0)

        self.weight = galsim.Image(
            bounds=self.coadd_bounds,
            wcs=self.coadd_wcs_bundle.galsim,
        )
        self.weight.fill(0)

    def setup_coadd_metacal(self, types):
        """
        Setup the coadd image and weight.

        Args:
            types (list): Metacal types in ["1m", "1p", "2m", "2p", "noshear"]
        """

        self.image = {}
        self.noise = {}
        self.weight = {}
        for type in types:
            self.image[type] = galsim.Image(
                bounds=self.coadd_bounds,
                wcs=self.coadd_wcs_bundle.galsim,
            )
            # Initially the image is fill with zeros to avoid issues in case
            # the exposures does not fill the entire footprint.
            self.image[type].fill(0)

            self.noise[type] = galsim.Image(
                bounds=self.coadd_bounds,
                wcs=self.coadd_wcs_bundle.galsim,
            )
            # Initially the noise is fill with zeros to avoid issues in case
            # the exposures does not fill the entire footprint.
            self.noise[type].fill(0)

            self.weight[type] = galsim.Image(
                bounds=self.coadd_bounds,
                wcs=self.coadd_wcs_bundle.galsim,
            )
            self.weight[type].fill(0)
