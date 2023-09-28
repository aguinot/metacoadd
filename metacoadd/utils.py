from typing import Optional, Union

import astropy.wcs
import galsim
import ngmix
import numpy as np


class WCSBundle:
    _astropy_wcs: astropy.wcs.WCS
    _galsim_wcs: galsim.BaseWCS

    def __init__(
        self,
        wcs: Union[astropy.wcs.WCS, galsim.BaseWCS],
        image: Optional[Union[np.ndarray, galsim.Image]] = None,
    ):
        """Create a bundled wcs containing both astropy and galsim wcs.

        The provided wcs can be either an astropy wcs or a galsim wcs.
        If wcs is an astropy wcs: create a galsim.AstropyWCS (which is a
        BaseWCS) and keep both.

        If wcs is a galsim wcs:
            - image must be given and be either an ndarray or a galsim Image
            - the astropy wcs will be created from the galsim wcs and the image
              bounds (either using image.bounds if galsim image or
              BoundsI(1, 1 , nrow, ncol) if ndarray).

        The wcs can be accessed with properties wcs_bundle.astropy or
        wcs_bundle.galsim
        """
        if isinstance(wcs, astropy.wcs.WCS):
            self._astropy_wcs = wcs
            self._galsim_wcs = galsim.AstropyWCS(header=wcs.to_header())
        elif isinstance(wcs, galsim.BaseWCS):
            self._galsim_wcs = wcs
            # in this case, we need the image
            if isinstance(image, np.ndarray):
                # we have no wcs in this image
                xmin, ymin = 1, 1
                nrow, ncol = image.shape
                bounds = galsim.BoundsI(
                    xmin, xmin + ncol - 1, ymin, ymin + nrow - 1
                )
                header = {}
                wcs.writeToFitsHeader(header, bounds)
                self._astropy_wcs = astropy.wcs.WCS(header)
            elif isinstance(image, galsim.Image):
                # There may be a wcs in this image, check they are compatible
                if image.wcs is not None:
                    reset_wcs = False
                    if image.wcs != wcs:
                        raise ValueError(
                            "image wcs and provided wcs are different"
                        )
                else:
                    reset_wcs = True
                    image.wcs = wcs
                header = {}
                image.wcs.writeToFitsHeader(header, image.bounds)
                self._astropy_wcs = astropy.wcs.WCS(header)
                if reset_wcs:
                    image.wcs = None
            else:
                raise ValueError(
                    "missing image or wrong type, needed when giving galsim wcs"
                )
        else:
            raise TypeError("wcs must be a galsim.BaseWVS or astropy.wcs.WCS")

    @property
    def galsim(self) -> galsim.BaseWCS:
        return self._galsim_wcs

    @property
    def astropy(self) -> astropy.wcs.WCS:
        return self._astropy_wcs


def shift_wcs(wcs: WCSBundle, offset: galsim.PositionI) -> WCSBundle:
    """Create a new WCSBundle by shifting the origin of the wcs by given
    offset (the original wcs is not modified).
    """
    if not isinstance(wcs, WCSBundle):
        raise TypeError(f"expected a WCSBundle for wcs, got {type(wcs)}")
    if not isinstance(offset, galsim.PositionI):
        raise TypeError(f"expected PositionI for offset, got {type(offset)}")

    # Get header
    h = wcs.astropy.to_header(relax=True)
    orig_crpix1 = h["CRPIX1"]
    orig_crpix2 = h["CRPIX2"]

    # Shift center
    # NOTE: Only the pixel reference need to be chenged
    new_crpix1 = orig_crpix1 - offset.x + 1
    new_crpix2 = orig_crpix2 - offset.y + 1
    h["CRPIX1"] = new_crpix1
    h["CRPIX2"] = new_crpix2

    astropy_wcs = astropy.wcs.WCS(h)
    wcs_bundle = WCSBundle(astropy_wcs)

    return wcs_bundle


def exp2obs(exp, exp_psf=None, use_resamp=True):
    if use_resamp:
        kind = "_resamp"
    else:
        kind = ""
    # Set images
    if hasattr(exp, "image" + kind):
        img = getattr(exp, "image" + kind).array
    else:
        raise ValueError("Exposure has no image set.")

    if hasattr(exp, "weight" + kind):
        weight = getattr(exp, "weight" + kind).array
    else:
        weight = None

    if hasattr(exp, "noise" + kind):
        noise = getattr(exp, "noise" + kind).array
    else:
        noise = None

    # Set wcs
    wcs_bundle = getattr(exp, "wcs_bundle" + kind)

    if not isinstance(exp_psf, type(None)):
        if hasattr(exp_psf, "image" + kind):
            img_psf = getattr(exp_psf, "image" + kind).array + 1e-5
        else:
            raise ValueError("PSF Exposure has no image set.")

        if hasattr(exp_psf, "weight" + kind):
            getattr(exp_psf, "weight" + kind).array
        else:
            pass

        wcs_bundle_psf = getattr(exp_psf, "wcs_bundle" + kind)

    dim = np.array(img.shape)
    cen = (dim - 1) / 2.0
    img_jac = ngmix.Jacobian(
        x=cen[0],
        y=cen[1],
        wcs=wcs_bundle.galsim.jacobian(
            image_pos=galsim.PositionD(cen[0], cen[1]),
        ),
    )

    if not isinstance(exp_psf, type(None)):
        dim_psf = np.array(img_psf.shape)
        cen_psf = (dim_psf - 1) / 2.0
        psf_jac = ngmix.Jacobian(
            row=cen_psf[0],
            col=cen_psf[1],
            wcs=wcs_bundle_psf.galsim.jacobian(
                image_pos=galsim.PositionD(cen_psf[0], cen_psf[1]),
            ),
        )
        psf_obs = ngmix.Observation(
            image=img_psf,
            weight=None,
            jacobian=psf_jac,
        )

    obs = ngmix.Observation(
        image=img,
        weight=weight,
        noise=noise,
        jacobian=img_jac,
        ormask=np.zeros_like(
            img,
            dtype=np.int32,
        ),
        bmask=np.zeros_like(
            img,
            dtype=np.int32,
        ),
        psf=psf_obs,
    )

    return obs
