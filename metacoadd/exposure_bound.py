import copy

import galsim
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from metacoadd.utils import shift_wcs


class ExposureBound:
    """ExposureBound

    Structure to store all the information for an exposure bound.

    TODO: Add consistency check if several images are provided:
        Same size. Other?

    Args:
        header (astropy.io.fits.header.Header): Image header containing all
            the WCS information. Either header or wcs has to be provided, not
            both.
        wcs (galsim.BaseWCS or astropy.wcs.wcs.WCS): wcs corresponding to the
            images. Either header or wcs has to be provided, not both.
        meta (dict): Add metadata information in the form of a dictionary. For
            example, it can be used to store the exposure ID as follow:
            meta = {'ID': 12345}. Defaults to None.
    """

    def __init__(
        self,
        image_bounds,
        header=None,
        wcs=None,
        meta=None,
    ):
        self._exposure_bounds = []

        if header is not None:
            if wcs is not None:
                raise ValueError(
                    "Either header or wcs has to be provided, not both."
                )
            if isinstance(header, fits.header.Header):
                self.header = header
                # Set WCS
                # We do that first because we need it for consistency checks.
                self._set_wcs(header=header)
            else:
                raise TypeError(
                    "header must be an astropy.io.fits.header.Header."
                )
        elif wcs is not None:
            if isinstance(wcs, galsim.BaseWCS):
                self._set_wcs(galsim_wcs=wcs)
            elif isinstance(wcs, WCS):
                self._set_wcs(astropy_wcs=wcs)
            else:
                raise TypeError(
                    f"wcs must be a galsim.BaseWCS or {type(WCS)}."
                )
        else:
            raise ValueError("Either header or wcs has to be provided")

        self._init_input_image_bound(image_bounds)

        self._set_meta(meta)

    def __getitem__(self, bounds):
        """
        Return a new ExposureBound instance with the corresponding new
        boundaries.
        Also handle the WCS.

        Args:
            bounds (galsim.BoundsI): New bounds for the images.

        Returns:
            ExposureBound: a new ExposureBound instance.
        """
        if not isinstance(bounds, galsim.BoundsI):
            raise TypeError("bounds must be a galsim.BoundsI.")
        new_exp_dict = {}
        new_exp_dict["image_bounds"] = bounds

        # We need to update the WCS to match new origin
        # WARNING: only if the origin changes
        orig_wcs = copy.deepcopy(self.wcs)
        if self._meta["image_bounds"] != bounds:
            offset_wcs = galsim.PositionI(bounds.xmin, bounds.ymin)
            new_exp_dict["wcs"] = shift_wcs(orig_wcs, offset_wcs)
            new_exp_dict["image_bounds"].wcs = new_exp_dict["wcs"]
        else:
            # If same bounds we still run the shift but we set the shift to
            # the fits origin.
            new_exp_dict["wcs"] = shift_wcs(orig_wcs, galsim.PositionI(1, 1))
            new_exp_dict["image_bounds"].wcs = new_exp_dict["wcs"]

        new_exposure = ExposureBound(
            meta=copy.deepcopy(self._meta), **new_exp_dict
        )

        # The image_bounds is used for the coadding only, to keep track of the
        # original bounds of the image.
        # NOTE: might need to find a better way to handle this!
        new_exposure._meta["image_bounds"] = self._meta["image_bounds"]

        return new_exposure

    def _set_wcs(self, header=None, galsim_wcs=None, astropy_wcs=None):
        """Set WCS

        Set the WCS in galsim and astropy format. The WCS are initialize from
        an astropy.io.fits.header.Header or astropy.wcs.wcs.WCS.
        """
        if not isinstance(header, type(None)):
            astropy_wcs = WCS(header)
            self.wcs = galsim.AstropyWCS(header=astropy_wcs)
            self.wcs.astropy = astropy_wcs
        elif not isinstance(galsim_wcs, type(None)):
            self.wcs = galsim_wcs
        elif not isinstance(astropy_wcs, type(None)):
            self.wcs = galsim.AstropyWCS(wcs=astropy_wcs)
            self.wcs.astropy = astropy_wcs

    def _set_astropy_wcs(self, galsim_bound):
        """Set astropy WCS

        Convert galsim WCS to astropy. This can only be done once we have a
        galsim image.

        Args:
            galsim_image (galsim.Image): a galsim image.
        """

        h_tmp = fits.Header()
        # h_tmp is directly updated
        galsim_bound.wcs.writeToFitsHeader(h_tmp, galsim_bound)
        astropy_wcs = WCS(h_tmp)
        self.wcs.astropy = astropy_wcs
        galsim_bound.wcs.astropy = astropy_wcs

    def _set_galsim_bound(self, image_bounds):
        """Set GalSim bound

        Transform the input array of bounds as a galsim.BoundsI.
        Args:
            image_bounds (numpy.ndarray, list): List of bounds to transform to
            a galsim.BoundsI.
        Returns:
            galsim.BoundsI: The corresponding galsim.BoundsI.
        """

        if not hasattr(self, "wcs"):
            self._set_wcs()

        galsim_bound = galsim.BoundsI(
            xmin=image_bounds[0],
            xmax=image_bounds[1],
            ymin=image_bounds[2],
            ymax=image_bounds[3],
        )

        return galsim_bound

    def _init_input_image_bound(self, image_bounds):
        """Set input image

        Check if the input image is a valid input and add it to ExposureBound.

        Args:
            image (numpy.ndarray or galsim.Image): Image to setup.
        """

        if isinstance(image_bounds, np.ndarray) | isinstance(
            image_bounds, list
        ):
            galsim_bound = self._set_galsim_bound(image_bounds)
        elif isinstance(image_bounds, galsim.BoundsI):
            if not hasattr(self, "wcs"):
                self._set_wcs()
            image_bounds.wcs = self.wcs
            galsim_bound = image_bounds
        else:
            raise TypeError(
                "image must be a numpy.ndarray or list or a galsim.BoundsI. "
                f"Got {type(image_bounds)}."
            )
        self._exposure_bounds.append("image_bounds")
        setattr(self, "image_bounds", galsim_bound)

        # In case galsim WCS where provided as input we set now the astropy one
        # We need information that become available only once we have set the
        # galsim image
        if not hasattr(self.wcs, "astropy"):
            self._set_astropy_wcs(galsim_bound)

    def _set_meta(self, meta):
        """
        Set metadata information.
        At moment, save only the image bounds.

        Args:
            meta (dict): Metadata to add.
        """

        if meta is not None:
            if not isinstance(meta, dict):
                raise TypeError("meta must be a dictionary.")
            self._meta = meta
        else:
            self._meta = {}

        self._meta["image_bounds"] = self.image_bounds


class ExpBList(list):
    """ExposureBound list

    List of ExposureBound.
    """

    def __init__(self):
        super().__init__()

    def append(self, exp):
        """append

        Add a new ExposureBound to the list.

        Args:
            exp (metacoadd.ExposureBound): ExposureBound to add.
        """

        if not isinstance(exp, ExposureBound):
            raise TypeError("exp must be a metacoadd.ExposureBound.")
        super().append(exp)

    def __setitem__(self, index, exp):
        """[summary]

        Args:
            index ([type]): [description]
            exp ([type]): [description]
        """
        if not isinstance(exp, ExposureBound):
            raise TypeError("exp must be a metacoadd.ExposureBound.")
        super().__setitem__(index, exp)


class PrepCoaddBound:
    """PrepCoaddBound

    Structure to store all the information to prepare the coadd.
    This class do not build the coadd but will pre-compute the exposure
    boundaries within the coadd footprint.

    NOTE: This class is aimed at being used in simulation to only prepare the
    area of an exposure that will enter a coadd and avoid to simulate the
    entire exposure.

    Args:
        expblist (metacoadd.ExpBList): ExpBList object that store all the
            exposure bound to build the coadd. It can also include bound that
            do not contribute to the coadd area and they will be automatically
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
        resize_exposure (bool, optional): Whether to resize the exposures
            before doing the interpolation. It is recommended to leave this to
            `True` since it will save computing time and memory. We use a
            "relax" parameters to make the resizing slightly larger the the
            coadd size given that this operation happen before the
            interpolation. This avoid to cut a part of the exposure due to
            projection effect later. See `relax_resize`. This is not
            related to the padding for the interpolation. Defaults to True.
        relax_resize (float, optional): Default relax parameters for
            the resizing (see above). Correspond to a percentage of the coadd
            size for both axes. Has to be in ]0, 1] (no good reason to go for
            more than 1 given that distortion effect are small). This can be
            internally change in case we reach one of the border of the
            exposure. Default to 0.10.
    """

    def __init__(
        self,
        expblist,
        world_coadd_center,
        scale,
        image_coadd_size=None,
        world_coadd_size=None,
        resize_exposure=True,
        relax_resize=0.10,
    ):
        if isinstance(expblist, ExpBList):
            self._orig_expblist = expblist
        else:
            raise TypeError("expblist has to be a metacoadd.ExpBList.")

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

        # Set galsim bounds and derive center for the coadd
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
                    self.resize_expblist(relax_resize)
                    self._relax_resize = relax_resize
                else:
                    raise ValueError("relax_resize has to be in ]0, 1].")
            else:
                raise TypeError("relax_resize has to be a float.")
        else:
            self.resize_expblist(0)

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
        convenience.
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

        self.wcs = galsim.TanWCS(
            affine=affine_transform,
            world_origin=self.world_coadd_center,
            units=galsim.arcsec,
        )
        self._set_astropy_wcs()
        self.coadd_pixel_scale = scale

    def _set_astropy_wcs(self):
        """Set astropy WCS

        Convert galsim WCS to astropy. This can only be done once we have a
        galsim image.

        Args:
            galsim_image (galsim.Image): a galsim image.
        """

        h_tmp = fits.ImageHDU(np.zeros(self.image_coadd_size)).header
        # h_tmp is directly updated
        self.wcs.writeToFitsHeader(h_tmp, self.coadd_bounds)
        astropy_wcs = WCS(h_tmp)
        self.wcs.astropy = astropy_wcs

    def resize_expblist(self, relax_resize):
        """Resize ExposureBound list

        Args:
            relax_resize (float): Resize relax parameter.
        """

        resized_expblist = ExpBList()
        for expb in self._orig_expblist:
            resized_exp = self._resize_bound(expb, relax_resize)
            if resized_exp is not None:
                resized_expblist.append(resized_exp)

        if len(resized_expblist) == 0:
            raise ValueError(
                "None of the provided ExposureBound overlap with the coadd "
                "area."
            )
        else:
            self.expblist = resized_expblist

    def _resize_bound(self, expb, relax_resize):
        """Resize bound

        Args:
            exp (metacoadd.ExposureBound): ExposureBound to resize.
            relax_resize (float): Resize relax parameter.
        Returns:
            metacoadd.ExposureBound or `None`: Return the resized bound or None
                if the bound is not in the coadd footprint.
        """

        # Here we need to round the position of the coadd but this is just to
        # compute a rough footprint of the coadd on the exposure boundaries. We
        # will estimate this latter with a better accuracy.
        # NOTE: This approximation might lead to remove an exposure that was
        #       in the coadd footprint but it would have contribute for a few
        #       pixels (a line or a column maximum).
        try:
            image_coadd_center_on_exp = expb.wcs.toImage(
                self.world_coadd_center
            )
        except TypeError:
            world_pos = galsim.PositionD(
                self.world_coadd_center.ra.deg,
                self.world_coadd_center.dec.deg,
            )
            image_coadd_center_on_exp = expb.wcs.toImage(world_pos)

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
        # and the exposure bound
        overlap = expb.image_bounds & new_bounds
        if not overlap.isDefined():
            return None
        # Now check if the entire coadd footprint is within the exposure bound
        if expb.image_bounds.includes(new_bounds):
            return expb[new_bounds]
        # if not, we cut the coadd footprint at the exposure bound edges
        else:
            new_bounds = new_bounds & expb.image_bounds
            return expb[new_bounds]

    def make_images(self, do_coadd=False):
        """make_image

        Create galsim.Image for each bounds in ExpBList.
        """

        images = []
        for expb in self.expblist:
            img = galsim.Image(expb.image_bounds, wcs=expb.wcs)
            images.append(img)

        if do_coadd:
            coadd_img = galsim.Image(self.coadd_bounds, wcs=self.wcs)
            return images, coadd_img
        else:
            return images
