import copy

import galsim
import ngmix
import numpy as np


def shift_wcs(wcs, offset):
    # TODO: check inputs

    wcs_orig = copy.deepcopy(wcs)
    if hasattr(wcs_orig, "astropy"):
        ap_wcs = wcs_orig.astropy
    elif hasattr(wcs_orig, "wcs"):
        ap_wcs = wcs_orig.wcs
    else:
        raise ValueError(
            "wcs must have an astropy component. Either .astropy or .wcs"
        )

    # Get header
    h = ap_wcs.to_header(relax=True)
    orig_crpix1 = h["CRPIX1"]
    orig_crpix2 = h["CRPIX2"]

    # Shift center
    # NOTE: Only the pixel reference need to be chenged
    new_crpix1 = orig_crpix1 - offset.x + 1
    new_crpix2 = orig_crpix2 - offset.y + 1
    h["CRPIX1"] = new_crpix1
    h["CRPIX2"] = new_crpix2

    new_wcs = galsim.AstropyWCS(header=h)

    return new_wcs


def _exp2obs(exp, exp_psf=None, use_resamp=False):
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
    wcs = getattr(exp, "wcs" + kind)

    if not isinstance(exp_psf, type(None)):
        if hasattr(exp_psf, "image" + kind):
            img_psf = getattr(exp_psf, "image" + kind).array + 1e-5
        else:
            raise ValueError("PSF Exposure has no image set.")

        if hasattr(exp_psf, "weight" + kind):
            getattr(exp_psf, "weight" + kind).array
        else:
            pass

        wcs_psf = getattr(exp_psf, "wcs" + kind)

    dim = np.array(img.shape)
    cen = (dim - 1) / 2.0
    img_jac = ngmix.Jacobian(
        x=cen[0],
        y=cen[1],
        wcs=wcs.jacobian(
            image_pos=galsim.PositionD(cen[0], cen[1]),
        ),
    )

    if not isinstance(exp_psf, type(None)):
        dim_psf = np.array(img_psf.shape)
        cen_psf = (dim_psf - 1) / 2.0
        psf_jac = ngmix.Jacobian(
            row=cen_psf[0],
            col=cen_psf[1],
            wcs=wcs_psf.jacobian(
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
