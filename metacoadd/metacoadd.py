

import galsim
import numpy as np
from .exposure import Exposure, CoaddImage


_available_method = ['weighted']


class SimpleCoadd():
    """SimpleCoadd

    This class handle the coaddition of an `Exposure` list.
    It will transform the coordinates and shifts each single exposures to match
    the coadd center.
    At the moment only weithed average coadding is handled.

    Args:
        coaddimage (metacoadd.CoaddImage): CoaddImage instance to stack.
        coadd_method (str, optional): Kind of stacking method to use. Only
        'weighted' is implemented. Defaults to 'weighted'.
            - `'weighted'`: Weighted average coadd.
    """

    def __init__(
        self,
        coaddimage,
        coadd_method='weighted',
    ):

        if isinstance(coaddimage, CoaddImage):
            self.coaddimage = coaddimage
        else:
            raise TypeError('coaddimage must be a metacoadd.CoaddImage.')

        if isinstance(coadd_method, str):
            if coadd_method in _available_method:
                self._coadd_method = coadd_method
            else:
                raise ValueError(
                    f'coadd_method must be in {_available_method}.'
                )
        else:
            raise TypeError('coadd_method must be of type str.')

        # NOTE: Not sure if this should be accessible by the user..
        self._border_size = 10

        self._set_coadd_pixel()

    def go(self):
        """
        Run the coaddition process.
        """

        if len(self.coaddimage.explist) == 0:
            raise ValueError('No exposure find to make the coadd.')
        if not self.coaddimage.explist[0]._interp:
            raise ValueError('Exposure must be interpolated first.')

        self.coaddimage.setup_coadd()
        stamps = []
        for exp in self.coaddimage.explist:
            all_stamp = self._process_one_exp(exp)

            # Check bounds, it should always pass. Just for safety.
            # We check only 'image' because it we always be there and the
            # property are shared with the other kind.
            b = all_stamp['image'].bounds & self.coaddimage.image.bounds
            if b.isDefined():
                if self._coadd_method == "weighted":
                    # NOTE: check for the presence of a 'weight' for the
                    # weighted average coadding
                    self.coaddimage.image[b] += \
                        all_stamp['image'] \
                        * all_stamp['weight'] \
                        * all_stamp['border']
                    if 'noise' in list(all_stamp.keys()):
                        self.coaddimage.noise[b] += \
                            all_stamp['noise'] \
                            * all_stamp['weight'] \
                            * all_stamp['border']
                    self.coaddimage.weight += \
                        all_stamp['weight'] \
                        * all_stamp['border']
            stamps.append(all_stamp)
        self.stamps = stamps
        non_zero_weights = np.where(self.coaddimage.weight.array != 0)
        self.coaddimage.image.array[non_zero_weights] /= \
            self.coaddimage.weight.array[non_zero_weights]
        if 'noise' in list(all_stamp.keys()):
            self.coaddimage.noise.array[non_zero_weights] /= \
                self.coaddimage.weight.array[non_zero_weights]

    def _process_one_exp(self, exp):
        """Process one exposure

        Make the coadding step for one exposure (deconv, reconv, ...).
        NOTE: find a better way to do this.

        Args:
            exp (metacoadd): Exposure to coadd.
        Returns:
            (dict): Dict containing all the images link to an exposure in the
                coadd referential.
        """

        if not isinstance(exp, Exposure):
            raise TypeError('exp must be a metacoadd.Exposure.')

        stamp_dict = {}

        # Process the image with pixel deconvolution. This is not optimal but
        # it links with the final goal where we will deconvolve by the PSF
        # instead.
        stamp_dict['image'] = self._change_pixel_deconv(exp, 'image')
        # stamp_dict['image'] = self._change_pixel_aff_trans(exp, 'image')
        # The noise image goes through the same process.
        if hasattr(exp, 'noise'):
            stamp_dict['noise'] = self._change_pixel_deconv(exp, 'noise')

        # Process the weight and/or flag with "classic" approach using affine
        # transformation
        if hasattr(exp, 'weight'):
            stamp_dict['weight'] = self._change_pixel_aff_trans(exp, 'weight')

        # Deal with the border.
        # If an exposure has been interpolated near the border it can creates
        # artifacts. To avoid propagating that into the coadd we remove a fixed
        # amount of pixels around the exposure. This has to be done at this
        # stage because the bordure of an exposure can fall in the midle of the
        # coadd.
        exp_offset = self._get_exp_offset(exp)
        border_image = self._set_border(exp)
        border_stamp = border_image.drawImage(
                bounds=self.coaddimage.coadd_bounds,
                center=exp.image.center,
                offset=exp_offset,
                wcs=self.coaddimage.coadd_wcs,
                method='no_pixel',
            )
        stamp_dict['border'] = border_stamp

        return stamp_dict

    def _change_pixel_deconv(self, exp, image_kind):
        """
        Change the pixel using deconvolution/convolution.

        Args:
            exp (metacoadd.Exposure): Exposure to transform.
            image_kind (str): The to which image in `exp` we apply the
                transformation

        Returns:
            galsim.Image: The new image with the coadd pixel.
        """

        # Some global informations
        image_pixel = self._get_exposures_pixel(exp)
        exp_offset = self._get_exp_offset(exp)

        original_image = getattr(exp, image_kind+'_interp')

        # Deconvolve from the original pixel
        deconv_img = self.deconvolve(
            original_image,
            image_pixel
        )

        # Convolve by new pixel
        new_img = galsim.Convolve((deconv_img, self._coadd_pixel))

        # Draw image
        image_stamp = new_img.drawImage(
            bounds=self.coaddimage.coadd_bounds,
            center=exp.image.center,
            offset=exp_offset,
            wcs=self.coaddimage.coadd_wcs,
            method='no_pixel',
        )

        return image_stamp

    def _change_pixel_aff_trans(self, exp, image_kind):
        """
        Change the pixel using affine transform.

        Args:
            exp (metacoadd.Exposure): Exposure to transform.
            image_kind (str): The to which image in `exp` we apply the
                transformation

        Returns:
            galsim.Image: The new image with the coadd pixel.
        """

        # Some global informations
        image_jacobian = exp.wcs.jacobian(
            world_pos=self.coaddimage.world_coadd_center,
        )
        image_jacobian_inv = image_jacobian.inverse()
        exp_offset = self._get_exp_offset(exp)
        coadd_jacobian = self.coaddimage.coadd_wcs.jacobian(
            world_pos=self.coaddimage.world_coadd_center,
        )

        original_image = getattr(exp, image_kind+'_interp')

        # Remove current pixel transformation
        deconv_img = galsim.Transform(
            original_image,
            jac=image_jacobian_inv.getMatrix().ravel(),
        )

        # Apply coadd pixel
        new_img = galsim.Transform(
            deconv_img,
            jac=coadd_jacobian.getMatrix().ravel(),
        )

        image_stamp = new_img.drawImage(
            bounds=self.coaddimage.coadd_bounds,
            center=exp.image.center,
            offset=exp_offset,
            wcs=self.coaddimage.coadd_wcs,
            method='no_pixel',
        )

        return image_stamp

    def _set_coadd_pixel(self):
        """
        Set the pixel information for the coadd image.
        """

        pixel = galsim.Pixel(scale=1)
        self._coadd_pixel = self.coaddimage.coadd_wcs.toWorld(
            pixel,
            world_pos=self.coaddimage.world_coadd_center,
        )

    def _get_exposures_pixel(self, exp):
        """
        Get the pixel information for the all the single exposure images.

        Args:
            exp (metacoadd.Exposure): Exposure from which we want the pixel
                information.
        Returns
            galsim.transform.Transform: Pixel transformation.
        """

        pixel = galsim.Pixel(scale=1)

        if not exp.image.wcs.isLocal():
            exp_pixel = exp.image.wcs.toWorld(
                pixel,
                world_pos=self.coaddimage.world_coadd_center,
            )
        else:
            exp_pixel = exp.image.wcs.toWorld(
                pixel,
                world_pos=galsim.PositionD(
                    self.coaddimage.world_coadd_center.ra.deg,
                    self.coaddimage.world_coadd_center.dec.deg,
                ),
            )

        return exp_pixel

    def deconvolve(self, image, psf):
        """Deconvolve

        Deconvolve from the pixel/psf information.
        NOTE: Find an other name for "psf"

        Args:
            image (galsim.Image): Image to deconvolve.
            psf (galsim.GSObject or galsim.transformation.Transformation):
                Galsim object or transformation to use for the deconvolution.
        Returns:
            no_psf (galsim.transformation.Transformation): Image deconvolve
                from the psf.
        """

        inv_psf = galsim.Deconvolve(psf)
        no_psf = galsim.Convolve((image, inv_psf))

        return no_psf

    def _get_exp_offset(self, exp):
        """Get exposure shift

        We compute the necessary offset to draw the exposure in the coadd
        frame.

        Args:
            exp (metacodd.Exposure): EXposure to shift.

        Returns:
            galsim.position.PositionD: Offset to apply in drawImage.
        """

        # Here we create a fake WCS object as if the entire single exposure had
        # the same WCS of the coadd. We need to do that to compute where is the
        # center of the coadd once the single image will be resampled using the
        # coadd WCS. We need that to draw the resample single exposure at the
        # right position.
        affine_transform = galsim.AffineTransform(
            self.coaddimage.coadd_pixel_scale,
            0.,
            0.,
            self.coaddimage.coadd_pixel_scale,
            origin=exp.image.center,
        )
        tmp_wcs = galsim.TanWCS(
            affine=affine_transform,
            world_origin=exp.wcs.toWorld(exp.image.center)
        )
        coadd_center_on_exp = tmp_wcs.toImage(
            self.coaddimage.world_coadd_center
        )

        # The (-) signe is here because drawImage will substract the offset.
        # We first shift to center of the coadd and then we substract the
        # center of the coadd (in the coadd reference) because the image in
        # GalSim are define in the corner.
        offset = -coadd_center_on_exp + self.coaddimage.image_coadd_center

        return offset

    def _set_border(self, exp):
        """Set border

        This method handle the CCD border to avoid issues in case the edge of
        a CCD falls in the coadd footprint.
        This step is necessarry due to the interpolation.

        Args:
            exp (metacoadd.Exposure): Exposure for which we create the border.

        Returns:
            galsim.Image: Galsim.Image representing te border.
        """

        full_bounds = exp._meta['image_bounds']

        # Here we can directly set to the coadd wcs.
        # Actually we create a WCS as if the exposure has the WCS
        # transformation of the coadd.
        # NOTE: not very "elegant" to do it this way.
        affine_transform = galsim.AffineTransform(
            self.coaddimage.coadd_pixel_scale,
            0,
            0,
            self.coaddimage.coadd_pixel_scale,
            origin=exp.image.center,
        )
        border_wcs = galsim.TanWCS(
            affine=affine_transform,
            world_origin=exp.wcs.toWorld(exp.image.center),
        )
        border_wcs = exp.wcs

        border_image = galsim.Image(
            bounds=full_bounds,
            wcs=border_wcs,
        )
        border_image.fill(0)

        border_bounds = galsim.BoundsI(
            xmin=full_bounds.xmin + self._border_size,
            xmax=full_bounds.xmax - self._border_size,
            ymin=full_bounds.ymin + self._border_size,
            ymax=full_bounds.ymax - self._border_size,
        )
        border_image[border_bounds].fill(1)

        border_exp = Exposure(border_image, wcs=border_wcs)
        if self.coaddimage._relax_resize is None:
            resized_border = self.coaddimage._resize_exp(border_exp, 0)
        else:
            relax_resize = self.coaddimage._relax_resize
            resized_border = self.coaddimage._resize_exp(
                border_exp,
                relax_resize,
            )

        border_interp = self.coaddimage._do_interp(
            resized_border.image,
            border_exp.wcs,
            'nearest',
        )
        self.border_interp = border_interp

        return border_interp
