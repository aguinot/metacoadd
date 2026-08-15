import galsim
import ngmix
import numpy as np


def shift_wcs(wcs, offset):
    """Shift the WCS by a given offset.

    Parameters
    ----------
    wcs : galsim.wcs
        The WCS to shift.
    offset : galsim.PositionD
        The offset to apply to the WCS.

    Returns
    -------
    new_wcs : galsim.wcs
        The shifted WCS.

    """
    # TODO: check inputs

    if hasattr(wcs, "astropy"):
        ap_wcs = wcs.astropy
    elif hasattr(wcs, "wcs"):
        ap_wcs = wcs.wcs
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
    kind = "_resamp" if use_resamp else ""

    image_name = "image" + kind
    weight_name = "weight" + kind
    noise_name = "noise" + kind
    wcs_name = "wcs" + kind

    if not hasattr(exp, image_name):
        raise ValueError("Exposure has no image set.")

    image = getattr(exp, image_name)
    img = image.array

    weight = (
        getattr(exp, weight_name).array if hasattr(exp, weight_name) else None
    )

    noise = getattr(exp, noise_name).array if hasattr(exp, noise_name) else None

    wcs = getattr(exp, wcs_name)

    # ngmix uses zero-based array coordinates. The GalSim WCS must be evaluated
    # at the GalSim image coordinate.
    dim = np.asarray(img.shape)
    cen = (dim - 1) / 2.0

    img_jac = ngmix.Jacobian(
        row=cen[0],
        col=cen[1],
        wcs=wcs.jacobian(
            image_pos=image.true_center,
        ),
    )

    psf_obs = None

    if exp_psf is not None:
        if not hasattr(exp_psf, image_name):
            raise ValueError("PSF Exposure has no image set.")

        psf_image = getattr(exp_psf, image_name)
        img_psf = psf_image.array

        if hasattr(exp_psf, weight_name):
            weight_psf = getattr(exp_psf, weight_name).array
        else:
            weight_psf = None

        wcs_psf = getattr(exp_psf, wcs_name)

        dim_psf = np.asarray(img_psf.shape)
        cen_psf = (dim_psf - 1) / 2.0

        psf_jac = ngmix.Jacobian(
            row=cen_psf[0],
            col=cen_psf[1],
            wcs=wcs_psf.jacobian(
                image_pos=psf_image.true_center,
            ),
        )

        psf_obs = ngmix.Observation(
            image=img_psf,
            weight=weight_psf,
            jacobian=psf_jac,
        )

    return ngmix.Observation(
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


def atleast_mbobs(obs):
    """Convert an ngmix.Observation, ngmix.ObsList or ngmix.MultiBandObsList to
    a ngmix.MultiBandObsList.

    Parameters
    ----------
    obs : ngmix.Observation, ngmix.ObsList or ngmix.MultiBandObsList
        The observation(s) to convert.

    Returns
    -------
    mbobs : ngmix.MultiBandObsList
        The converted MultiBandObsList.

    """
    if isinstance(obs, ngmix.Observation):
        mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)
    elif isinstance(obs, ngmix.ObsList):
        mbobs = ngmix.MultiBandObsList()
        mbobs.append(obs)
    elif isinstance(obs, ngmix.MultiBandObsList):
        mbobs = obs
    else:
        raise ValueError(
            "obs must be an Observation, ObsList or MultiBandObsList"
        )

    return mbobs


def _identity_njit(*jit_args, **jit_kwargs):
    """Return the original Python function without compiling it."""
    if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
        return jit_args[0]

    def decorator(func):
        return func

    return decorator
