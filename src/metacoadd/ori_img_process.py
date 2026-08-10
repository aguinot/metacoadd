import numpy as np

import ngmix

from .detect import get_stamp_mbobs
from .fitting import get_gauss_psf_runner


def get_original_cat_dtype(nband):
    """Get the dtype for the original image catalog.

    Parameters
    ----------
    nband : int
        The number of bands.

    Returns
    -------
    dtype : list
        The dtype for the original image catalog.

    """
    dtype = []
    for i in range(nband):
        dtype += [
            (f"original_mxx_{i}", np.float64),
            (f"original_mxy_{i}", np.float64),
            (f"original_myy_{i}", np.float64),
            (f"original_weight_{i}", np.float64),
            (f"original_flux_{i}", np.float64),
            (f"original_flux_err_{i}", np.float64),
            (f"original_flags_{i}", np.int32),
            (f"original_psf_mxx_{i}", np.float64),
            (f"original_psf_mxy_{i}", np.float64),
            (f"original_psf_myy_{i}", np.float64),
            (f"original_psf_flags_{i}", np.int32),
        ]
    return dtype


def get_output_original_cat(n_obj, nband):
    """Get the output catalog for the original image measurements.

    Parameters
    ----------
    n_obj : int
        The number of objects.
    nband : int
        The number of bands.

    Returns
    -------
    out : np.ndarray
        The output catalog for the original image measurements.

    """
    CAT_DTYPE = get_original_cat_dtype(nband)
    out = np.array(
        list(map(tuple, np.zeros((len(CAT_DTYPE), n_obj)).T)),
        dtype=CAT_DTYPE,
    )
    return out


def get_original_image_cat(
    rng,
    mbobs,
    sep_cat,
    seg_map=None,
    cutout_size=101,
    do_uberseg=False,
    wmom_fwhm=0.5,
    psf_fitter="gauss",
):
    """Get the original image catalog.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    mbobs : MultiBandObsList
        The multi-band observations.
    sep_cat : np.ndarray
        The separation catalog.
    seg_map : np.ndarray, optional
        The segmentation map.
    cutout_size : int, optional
        The size of the cutout.
    do_uberseg : bool, optional
        Whether to do uber segmentation.
    wmom_fwhm : float, optional
        The FWHM for the weighted moment fitter.
    psf_fitter : str, optional
        The PSF fitter to use.

    Returns
    -------
    output_cat : np.ndarray
        The output catalog for the original image measurements.

    """
    output_cat = get_output_original_cat(len(sep_cat), len(mbobs))
    wmom_runner = ngmix.gaussmom.GaussMom(fwhm=wmom_fwhm)

    # Get PSF first
    if psf_fitter == "wmom":
        psf_runner = wmom_runner
    elif psf_fitter == "gauss":
        psf_runner = get_gauss_psf_runner(rng)
    else:
        raise ValueError(
            f"PSF fitter {psf_fitter} not recognized. Must be in "
            "['wmom', 'gauss']."
        )
    all_psf_res = []
    for obslist in mbobs:
        psf_res = psf_runner.go(obslist[0].psf)
        if psf_fitter == "wmom":
            myy, mxy, mxx = ngmix.moments.e2mom(
                psf_res["e1"], psf_res["e2"], psf_res["T"]
            )
        else:
            myy, mxy, mxx = ngmix.moments.g2mom(
                psf_res["g"][0], psf_res["g"][1], psf_res["T"]
            )
        all_psf_res.append(np.array([mxx, mxy, myy, psf_res["flags"]]))

    # Get the objects
    for obj_ind, det_obj in enumerate(sep_cat):
        obj_mb_obs = get_stamp_mbobs(
            mbobs,
            det_obj,
            min_stamp_size=cutout_size,
            max_stamp_size=cutout_size,
            do_uberseg=do_uberseg,
            seg_map=seg_map,
        )

        for band_i, obslist in enumerate(obj_mb_obs):
            res = wmom_runner.go(obslist[0])
            myy, mxy, mxx = ngmix.moments.e2mom(res["e1"], res["e2"], res["T"])
            wght = np.median(obslist[0].weight[obslist[0].weight > 0])
            output_cat[obj_ind][f"original_mxx_{band_i}"] = mxx
            output_cat[obj_ind][f"original_mxy_{band_i}"] = mxy
            output_cat[obj_ind][f"original_myy_{band_i}"] = myy
            output_cat[obj_ind][f"original_weight_{band_i}"] = wght
            output_cat[obj_ind][f"original_flux_{band_i}"] = res["flux"]
            output_cat[obj_ind][f"original_flux_err_{band_i}"] = res["flux_err"]
            output_cat[obj_ind][f"original_flags_{band_i}"] = res["flags"]
            output_cat[obj_ind][f"original_psf_mxx_{band_i}"] = all_psf_res[
                band_i
            ][0]
            output_cat[obj_ind][f"original_psf_mxy_{band_i}"] = all_psf_res[
                band_i
            ][1]
            output_cat[obj_ind][f"original_psf_myy_{band_i}"] = all_psf_res[
                band_i
            ][2]
            output_cat[obj_ind][f"original_psf_flags_{band_i}"] = all_psf_res[
                band_i
            ][3]

    return output_cat
