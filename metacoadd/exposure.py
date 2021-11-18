

import numpy as np
import galsim
from astropy.io import fits

from tqdm import tqdm


DEFAULT_INTERP_CONFIG = {
    'classic': {
        'pad_factor': 4,
        'x_interpolant': 'quintic',
        'k_interpolant': 'quintic',
        'calculate_maxk': True,
        'calculate_stepk': True,
    },
    'weight': {
        'pad_factor': 4,
        'x_interpolant': 'linear',
        'k_interpolant': 'linear',
        'calculate_maxk': True,
        'calculate_stepk': True,
    },
    'flag': {
        'pad_factor': 4,
        'x_interpolant': 'linear',
        'k_interpolant': 'linear',
        'calculate_maxk': True,
        'calculate_stepk': True,
    },
}

DEFAULT_GSPARAMS = galsim.GSParams(maximum_fft_size=8192)


class Exposure():
    """Exposure

    Structure to store all the informations for an exposure.

    Args:
        image (numpy.ndarray or galsim.Image): Science image.
        header (astropy.io.fits.header.Header): Image header containing all
            the WCS information. Either header or wcs has to be provided, not
            both.
        wcs (galsim.BaseWCS): wcs corresponding to the images. Either header or
            wcs has to be provided, not both.
        weight (numpy.ndarray or galsim.Image, optional): Weight image.
            Defaults to None.
        flag (numpy.ndarray or galsim.Image, optional): Flag image. Defaults to
            None.
        noise (numpy.ndarray or galsim.Image, optional): Noise image. Defaults
            to None.
        interp_config (dict, optional): Set of parameters for the
            interpolation. If `None` use the default configuration. Defaults
            to None.
        gsparams (galsim.GSParams, optional): set the gsparams for the
            interpolation. If `None` use the default configuration. Defaults
            to None.
    """

    def __init__(
        self,
        image,
        header=None,
        wcs=None,
        weight=None,
        flag=None,
        noise=None,
        interp_config=None,
        gsparams=None,
    ):

        self._exposure_images = []

        if header is not None:
            if wcs is not None:
                raise ValueError(
                    'Either header or wcs has to be provided, not both.'
                )
            if isinstance(header, fits.header.Header):
                self.header = header
                # Set WCS
                # We do that first because we need it for consistency checks.
                self._set_wcs()
            else:
                raise TypeError(
                    'header must be an astropy.io.fits.header.Header.'
                    )
        elif wcs is not None:
            if isinstance(wcs, galsim.BaseWCS):
                self.wcs = wcs
            else:
                raise TypeError('wcs must be a galsim.BaseWCS.')
        else:
            raise ValueError(
                'Either header or wcs has to be provided'
            )

        self._init_input_image(image, 'image')
        if weight is not None:
            self._init_input_image(image, 'weight')
        if flag is not None:
            self._init_input_image(image, 'flag')
        if noise is not None:
            self._init_input_image(image, 'noise')

        if interp_config is None:
            self.interp_config = DEFAULT_INTERP_CONFIG
        else:
            if isinstance(interp_config, dict):
                self.interp_config = interp_config
            else:
                raise ValueError('interp_config must be a dict.')

        if gsparams is None:
            self._gsparams = DEFAULT_GSPARAMS
        else:
            if isinstance(interp_config, galsim.GSParams):
                self._gsparams = gsparams
            else:
                raise ValueError('gsparams must be a galsim.GSParams.')

    def __getitem__(self, bounds):
        """
        Return a new Exposure instance with the corresponding subimages.

        Args:
            bounds (galsim.BoundsI): New bounds for the images.

        Returns:
            Exposure: a new Exposure instance.
        """
        if not isinstance(bounds, galsim.BoundsI):
            raise TypeError('bounds must be a galsim.BoundsI.')
        new_exposure = {}
        for image_kind in self._exposure_images:
            new_exposure[image_kind] = getattr(self, image_kind)[bounds]
        # Here it is safe to request 'image' since it is a requiered input.
        new_exposure['wcs'] = new_exposure['image'].wcs

        return Exposure(**new_exposure)

    def get_all_interp_images(
        self,
        weight_interp_method='linear',
        flag_interp_method='linear',
    ):
        """
        Get all interpolated images.

        Args:
            flag_interp_method (str, optional): Method to use to interpolate
                flag image. Defaults to 'linear'.
            weight_interp_method (str, optional): Method to use to interpolate
                weight image. Defaults to 'linear'.
        """

        print("Interpolate image...")
        self.image_interp = self._do_interp(self.image)

        if hasattr(self, 'weight'):
            print("Interpolate weight...")
            self.weight_interp = self._do_interp(
                self.weight,
                x_interpolant=weight_interp_method,
                k_interpolant=weight_interp_method,
            )

        if hasattr(self, 'flag'):
            print("Interpolate flag...")
            self.flag_interp = self._do_interp(
                self.flag,
                x_interpolant=flag_interp_method,
                k_interpolant=flag_interp_method,
            )

        if hasattr(self, 'noise'):
            print("Interpolate noise...")
            self.weight_interp = self._do_interp(
                self.noise,
            )

    def _set_wcs(self):
        """Set WCS

        Set the WCS in galsim format. The WCS are initialize from an
        astropy.io.fits.header.Header.
        """

        self.wcs = galsim.AstropyWCS(header=self.header)

    def _set_galsim_image(self, image):
        """Set GalSim image

        Transform the input image array as a galsim.Image.
        Args:
            image (numpy.ndarray): Image to transform to a galsim.Image.
        Returns:
            galsim.Image: The corresponding galsim.Image.
        """

        if not hasattr(self, 'wcs'):
            self._set_wcs()

        galsim_image = galsim.Image(
            image,
            xmin=0,
            ymin=0,
            wcs=self.wcs,
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
                if not hasattr(self, 'wcs'):
                    self._set_wcs()
                image.wcs = self.wcs
            elif image.wcs != self.wcs:
                raise ValueError(
                    'Inconsistant WCS between galsim.Image({image_kind}) and '
                    'proveded header.'
                )
            galsim_image = image
        else:
            raise TypeError(
                'image must be a ngmix.ndarray or a galsim.Image. '
                f'Got {type(image)}.'
            )
        self._exposure_images.append(image_kind)
        setattr(self, image_kind, galsim_image)

    def _do_interp(self, image, **kwargs):
        """Run interpolation

        Interpolate an image using galsim and return a
        `galsim.interpolatedimage.InterpolatedImage`. This method use the wcs
        set in the initialization by default.

        Args:
            image (numpy.ndarray): Image to interpolate.

        Returns:
            galsim.interpolatedimage.InterpolatedImage: Interpolated image.
        """

        interp_config = self.interp_config.copy()
        interp_config.update(kwargs)

        galsim_image = galsim.Image(image, wcs=self.wcs, copy=True)
        interp_image = galsim.InterpolatedImage(
            galsim_image,
            wcs=self.wcs,
            gsparams=self._gsparams,
            **interp_config,
        )

        return interp_image


class ExpList(list):
    """Exposure list

    List of Exposure.
    """

    def __init__(self):
        super().__init__()

    def append(self, exp):
        """append

        Add a new Exposure to the list.

        Args:
            exp (metacoadd.Exposure): Exposure to add.
        """

        if not isinstance(exp, Exposure):
            raise TypeError('exp must be a metacoadd.Exposure.')
        super().append(exp)

    def __setitem__(self, index, exp):
        """[summary]

        Args:
            index ([type]): [description]
            exp ([type]): [description]
        """
        if not isinstance(exp, Exposure):
            raise TypeError('exp must be a metacoadd.Exposure.')
        super().__setitem__(index, exp)


class CoaddImage():
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
            in world coordinates. If a `galsim.angle.Angle` is provided, will
            assume the coadd to be square. Otherwise, has to be a `list` or
            `tuple` of `galsim.angle.Angle`. Either `image_coadd_size` or
            `world_coadd_size` as to be provided.
        interp_config (dict, optional): Set of parameters for the
            interpolation. If `None` use the default configuration. Defaults to
            None.
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
        explist,
        world_coadd_center,
        scale,
        image_coadd_size=None,
        world_coadd_size=None,
        interp_config=None,
        resize_exposure=True,
        relax_resize=0.10,
        gsparams=None,
    ):

        if isinstance(explist, ExpList):
            self._orig_explist = explist
        else:
            raise TypeError('explist has to be a metacoadd.ExpList.')

        if isinstance(world_coadd_center, galsim.celestial.CelestialCoord):
            self.world_coadd_center = world_coadd_center
        else:
            raise TypeError(
                'world_coadd_center has to be a '
                'galsim.celestial.CelestialCoord'
            )

        if image_coadd_size is not None:
            if world_coadd_size is not None:
                raise ValueError(
                    'Either image_coadd_center or world_coadd_center has to '
                    'be provided, not both.'
                )
            if isinstance(image_coadd_size, list) \
               or isinstance(image_coadd_size, tuple):
                if all(isinstance(n, int) for n in image_coadd_size):
                    self.image_coadd_size = list(image_coadd_size)
                else:
                    raise TypeError(
                        'image_coadd_size has to be a list or tuple of int.'
                    )
            elif isinstance(image_coadd_size, int):
                self.image_coadd_size = [image_coadd_size]*2
            else:
                raise TypeError(
                    'image_coadd_size has to be a list, tuple or int.'
                )
        elif world_coadd_size is not None:
            if isinstance(world_coadd_size, list) \
               or isinstance(world_coadd_size, tuple):
                if all(isinstance(n, galsim.angle.Angle)
                       for n in world_coadd_size):
                    self._set_image_coadd_size(list(world_coadd_size), scale)
                else:
                    raise TypeError(
                        'image_coadd_size has to be a list or tuple of int.'
                    )
            elif isinstance(world_coadd_size, galsim.angle.Angle):
                self._set_image_coadd_size([world_coadd_size]*2, scale)
            else:
                raise TypeError(
                    'world_coadd_size has to be a list, tuple or '
                    'galsim.angle.Angle.'
                )
        else:
            raise ValueError(
                'Either image_coadd_size or world_coadd_size has to be '
                'provided'
            )

        # Set galsim bounds and dervie center for the coadd
        self._set_coadd_bounds()
        self._set_image_coadd_center()

        # Set coadd WCS
        if isinstance(scale, float):
            self._set_coadd_wcs(scale)
        else:
            TypeError('scale has to be a float')

        # Resize the exposures if requested
        if resize_exposure:
            if relax_resize is None:
                raise ValueError(
                    'relax_resize has to be provided to resize exposure'
                )
            if isinstance(relax_resize, float):
                if relax_resize > 0. and relax_resize <= 1.:
                    self.resize_explist(relax_resize)
                else:
                    raise ValueError('relax_resize has to be in ]0, 1]')
            else:
                raise TypeError('relax_resize has to be a float.')
        else:
            self.resize_explist(0)

        if interp_config is None:
            self.interp_config = DEFAULT_INTERP_CONFIG
        else:
            if isinstance(interp_config, dict):
                self.interp_config = interp_config
            else:
                raise TypeError('interp_config must be a dict.')

        if gsparams is None:
            self._gsparams = DEFAULT_GSPARAMS
        else:
            if isinstance(interp_config, galsim.GSParams):
                self._gsparams = gsparams
            else:
                raise TypeError('gsparams must be a galsim.GSParams.')

    def _set_image_coadd_size(self, world_coadd_size, scale):
        """Set coadd size

        Set the size of the coadd in pixels from angle.

        Args:
            world_coadd_size (list): List of `galsim.angle.Angle`.
            scale (float): Coadd pixel scale
        """

        from math import ceil

        size_x = ceil((world_coadd_size[0]/galsim.arcsec)/scale)
        size_y = ceil((world_coadd_size[1]/galsim.arcsec)/scale)

        self.image_coadd_size = [size_x, size_y]

    def _set_coadd_bounds(self):
        """
        Create a galsim.Image that describe the coadd. This is just for
        convinience.
        """

        self.coadd_bounds = galsim.BoundsI(
            xmin=0,
            xmax=self.image_coadd_size[0]-1,
            ymin=0,
            ymax=self.image_coadd_size[1]-1,
        )

    def _set_image_coadd_center(self):
        """
        Set coadd center in pixel.
        """

        self.image_coadd_center = self.coadd_bounds.center

    def _set_coadd_wcs(self, scale):
        """Set coadd wcs

        Set the coadd WCS as TAN projection with the given pixel scale.

        Args:
            scale (float): Coadd pixel scale.
        """

        affine_transform = galsim.AffineTransform(
            scale, 0., 0., scale,
            origin=self.image_coadd_center,
        )

        self.coadd_wcs = galsim.TanWCS(
            affine=affine_transform,
            world_origin=self.world_coadd_center,
            units=galsim.arcsec,
        )

    def resize_explist(self, relax_resize):
        """Resize exposure list

        Args:
            relax_resize (float): Resize relax parameter.
        """

        resized_explist = ExpList()
        for exp in self._orig_explist:
            resized_exp = self._resize_exp(exp, relax_resize)
            if resized_exp is not None:
                # Might be weird to set this here, but this function is called
                # automatically and will need this parameter later.
                # NOTE: Maybe find a better way to do this..
                resized_exp._interp = False
                resized_explist.append(resized_exp)

        if len(resized_explist) == 0:
            raise ValueError(
                'None of the provided exposure overlap with the coadd area.'
            )
        else:
            self.explist = resized_explist

    def _resize_exp(self, exp, relax_resize):
        """Resize exposure

        Args:
            exp (metacoadd.Exposure): Exposure to resize.
            relax_resize (float): Resize relax parameter.
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
            ).round()
        except TypeError:
            world_pos = galsim.PositionD(
                self.world_coadd_center.ra,
                self.world_coadd_center.dec,
                )
            image_coadd_center_on_exp = exp.image.wcs.toImage(
                world_pos
            ).round()

        # Make raw bounds
        new_bounds = galsim.BoundsI(
            xmin=0,
            xmax=int(self.coadd_bounds.xmax*(1.+relax_resize)),
            ymin=0,
            ymax=int(self.coadd_bounds.ymax*(1.+relax_resize)),
        )

        # Shift the bounds to the correct position
        new_bounds = new_bounds.shift(
            image_coadd_center_on_exp-new_bounds.center
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
            if new_bounds.xmin < exp_bounds.xmin:
                new_bounds.xmin = exp_bounds.xmin
            if new_bounds.xmax > exp_bounds.xmax:
                new_bounds.xmax = exp_bounds.xmax
            if new_bounds.ymin < exp_bounds.ymin:
                new_bounds.ymin = exp_bounds.ymin
            if new_bounds.ymax > exp_bounds.ymax:
                new_bounds.ymax = exp_bounds.ymax
            return exp[new_bounds]

    def get_all_interp_images(
        self,
        **kargs,
    ):
        """
        Get all interpolated images.

        Args:
            flag_interp_method (str, optional): Method to use to interpolate
                flag image. Defaults to 'linear'.
            weight_interp_method (str, optional): Method to use to interpolate
                weight image. Defaults to 'linear'.
            kwargs: Any additionnal keywords arguments for
                galsim.InterpolatedImage.
        """

        _image_kinds = ['image', 'weight', 'flag', 'noise']
        for exp in tqdm(self.explist, total=len(self.explist)):
            # ggalsim.InterpolatedImage works with local WCS so we force it to
            # take the one at the center of the coadd for better accuracy.
            # This probably do not make a difference but it is more
            # "elegant" to do it :)
            wcs = exp.wcs.local(world_pos=self.world_coadd_center)
            for image_kind in _image_kinds:
                if not hasattr(exp, image_kind):
                    continue

                if image_kind == 'weight':
                    interp_method = 'weight'
                elif image_kind == 'flag':
                    interp_method = 'flag'
                else:
                    interp_method = 'classic'

                print(f"Interpolate {image_kind}...")
                exp.image_interp = self._do_interp(
                    getattr(exp, image_kind),
                    wcs,
                    interp_method,
                    **kargs,
                    )
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
                interpolation to 'linear' for the weight and flag images.
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
