import galsim
# from astropy.io.fits import Header
import ngmix
import metadetect as mdet
import metacoadd as mtc

import numpy as np
from math import ceil

import copy
# import os


TEST_METADETECT_CONFIG = {
    "model": "wmom",

    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'gauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
        # 'fixnoise': True,
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,
        # 'deblend_cont': 0.01,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel': [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    'meds': {
        'min_box_size': 32,
        'max_box_size': 32,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # needed for PSF symmetrization
    # 'psf': {
    #     'model': 'gauss',

    #     'ntry': 2,

    #     'lm_pars': {
    #         'maxfev': 2000,
    #         'ftol': 1.0e-5,
    #         'xtol': 1.0e-5,
    #     }
    # },

    # check for an edge hit
    'bmask_flags': 2**30,

    'nodet_flags': 2**0,
}


def run_simplecoadd(
    sim_data,
    seed,
):
    """Run metacoadd

    Args:
        explist (metacoadd.ExpList): List of input exposures.
        coadd_ra (float): RA position of the coadd center.
        coadd_dec (float): DEC position of the coadd center.
        coadd_scale (float): Pixel scale to use for the coadd.
        coadd_size (float): Size of the coadd (in arcmin).
        explist_psf (metacoadd.ExpList, optional): List of input PSF for each
            exposure at the coadd center. Defaults to None.

    Returns:
        metacoadd.SimpleCoadd or tuple: Return an instanciate
            `metacoadd.SimpleCoadd` object. Or a tuple of them, one for the
            images and one for the PSF.
    """

    n_epoch = len(sim_data['band_data']['i'])

    # Process images
    explist = mtc.ExpList()
    rng = np.random.RandomState(seed)
    for i in range(n_epoch):
        exp = mtc.Exposure(
            image=sim_data['band_data']['i'][i].image.array,
            weight=1/sim_data['band_data']['i'][i].variance.array,
            noise=rng.normal(
                size=sim_data['band_data']['i'][i].image.array.shape
            )*np.sqrt(sim_data["band_data"]["i"][i].variance.array),
            wcs=copy.deepcopy(sim_data['band_data']['i'][i].wcs),
        )
        explist.append(exp)

    coaddimage = mtc.CoaddImage(
        explist,
        world_coadd_center=sim_data["coadd_wcs"].center,
        scale=np.abs(sim_data["coadd_wcs"].cd[0, 0]*3600),
        image_coadd_size=sim_data["coadd_dims"][0],
        resize_exposure=True,
        relax_resize=0.5,
    )
    coaddimage.get_all_resamp_images()

    simplecoadd = mtc.SimpleCoadd(coaddimage)
    simplecoadd.go()
    output = (simplecoadd, )

    # Process PSFs
    explist_psf = mtc.ExpList()
    for i in range(n_epoch):
        coadd_center_on_exp = sim_data['band_data']['i'][i].wcs.toImage(
                sim_data["coadd_wcs"].center
            )
        gs_img = sim_data['band_data']['i'][i].psf._get_gspsf(
            coadd_center_on_exp
        )
        psf_img_local = gs_img.drawImage(
            center=coadd_center_on_exp,
            nx=sim_data["psf_dims"][0],
            ny=sim_data["psf_dims"][1],
            wcs=copy.deepcopy(sim_data['band_data']['i'][i].wcs),
        )
        psf_weight = np.ones_like(psf_img_local.array) / 1e-5**2
        psf_weight_galsim = galsim.Image(
                psf_weight,
                bounds=psf_img_local.bounds,
            )
        exp_psf = mtc.Exposure(
            image=psf_img_local,
            weight=psf_weight_galsim,
            wcs=copy.deepcopy(sim_data['band_data']['i'][i].wcs),
        )
        explist_psf.append(exp_psf)

    coaddimage_psf = mtc.CoaddImage(
        explist_psf,
        world_coadd_center=sim_data["coadd_wcs"].center,
        scale=np.abs(sim_data["coadd_wcs"].cd[0, 0]*3600),
        image_coadd_size=sim_data["psf_dims"][0],
        resize_exposure=True,
        relax_resize=0.5
    )
    coaddimage_psf.get_all_resamp_images()

    simplecoadd_psf = mtc.SimpleCoadd(coaddimage_psf, do_border=False)
    simplecoadd_psf.go()
    output += (simplecoadd_psf, )

    return output


def run_metacoadd(
    explist,
    coadd_ra,
    coadd_dec,
    coadd_scale,
    coadd_size,
    explist_psf=None,
):
    """Run metacoadd

    Args:
        explist (metacoadd.ExpList): List of input exposures.
        coadd_ra (float): RA position of the coadd center.
        coadd_dec (float): DEC position of the coadd center.
        coadd_scale (float): Pixel scale to use for the coadd.
        coadd_size (float): Size of the coadd (in arcmin).
        explist_psf (metacoadd.ExpList, optional): List of input PSF for each
            exposure at the coadd center. Defaults to None.

    Returns:
        metacoadd.SimpleCoadd or tuple: Return an instanciate
            `metacoadd.SimpleCoadd` object. Or a tuple of them, one for the
            images and one for the PSF.
    """

    # Process images
    coaddimage = mtc.CoaddImage(
        explist=explist,
        world_coadd_center=galsim.CelestialCoord(
            ra=coadd_ra*galsim.degrees,
            dec=coadd_dec*galsim.degrees,
        ),
        scale=coadd_scale,
        image_coadd_size=ceil(coadd_size*3600/coadd_scale),
    )
    coaddimage.get_all_interp_images()

    psfs = [exp.image.array for exp in explist_psf]

    mc = mtc.MetaCoadd(coaddimage, psfs)
    mc.go()

    return mc


def run_metadetect(
    simplecoadd,
    simplecoadd_psf,
    seed,
):
    """Run metadetect

    Args:
        simplecoadd (metacoadd.SimpleCoadd): Coadd image.
        simplecoadd_psf (metacoadd.SimpleCoadd): Coadd PSF.
        seed (int): RNG seed.

    Returns:
        [type]: Output of `metadetect`.
    """

    # Make PSF
    # Here this is simple since all exposures share the same PSF
    # NOTE: Update this for a more general case
    psf_dim = simplecoadd_psf.coaddimage.image.array.shape[0]
    psf_cen = (psf_dim-1)/2.
    psf_jac = ngmix.DiagonalJacobian(
        scale=simplecoadd.coaddimage.coadd_pixel_scale,
        row=psf_cen,
        col=psf_cen,
    )

    # Coadd jacobian
    dim = simplecoadd.coaddimage.image.array.shape[0]
    cen = (dim-1)/2
    coadd_jac = ngmix.DiagonalJacobian(
        scale=simplecoadd.coaddimage.coadd_pixel_scale,
        row=cen,
        col=cen
    )

    # Make ngmix Obs
    obs = ngmix.Observation(
        image=simplecoadd.coaddimage.image.array,
        weight=simplecoadd.coaddimage.weight.array,
        noise=simplecoadd.coaddimage.noise.array,
        jacobian=coadd_jac,
        ormask=np.zeros_like(
            simplecoadd.coaddimage.image.array,
            dtype=np.int32
        ),
        bmask=np.zeros_like(
            simplecoadd.coaddimage.image.array,
            dtype=np.int32
        ),
        psf=ngmix.Observation(
            image=simplecoadd_psf.coaddimage.image.array,
            jacobian=psf_jac),
    )
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    res = mdet.do_metadetect(
        TEST_METADETECT_CONFIG,
        mbobs,
        np.random.RandomState(seed=seed),
    )

    return res


def run_metadetect_perfect(
    coadd,
    scale,
    seed,
):
    """Run metadetect

    Args:
        simplecoadd (metacoadd.SimpleCoadd): Coadd image.
        simplecoadd_psf (metacoadd.SimpleCoadd): Coadd PSF.
        seed (int): RNG seed.

    Returns:
        [type]: Output of `metadetect`.
    """

    rng = np.random.RandomState(seed)

    # Coadd jacobian
    dim = coadd.image.array.shape[0]
    cen = (dim-1)/2
    coadd_jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=cen,
        col=cen,
    )

    # Make PSF
    # Here this is simple since all exposures share the same PSF
    # NOTE: Update this for a more general case
    psf_dim = 51
    psf_cen = (psf_dim-1)/2.
    psf_image = coadd.psf.drawImage(
        nx=psf_dim,
        ny=psf_dim,
        wcs=coadd.wcs.local(image_pos=galsim.PositionD(cen, cen)),
    )
    psf_jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=psf_cen,
        col=psf_cen,
    )

    # Make ngmix Obs
    coadd_noise = rng.normal(
        size=coadd.image.array.shape
    )*np.sqrt(coadd.variance.array)
    obs = ngmix.Observation(
        image=coadd.image.array,
        weight=1./coadd.variance.array,
        noise=coadd_noise,
        jacobian=coadd_jac,
        ormask=np.zeros_like(
            coadd.image.array,
            dtype=np.int32
        ),
        bmask=np.zeros_like(
            coadd.image.array,
            dtype=np.int32
        ),
        psf=ngmix.Observation(
            image=psf_image.array,
            jacobian=psf_jac),
    )
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    res = mdet.do_metadetect(
        TEST_METADETECT_CONFIG,
        mbobs,
        np.random.RandomState(seed=seed),
    )

    return res


def _shear_cuts(arr, model):
    if model == "wmom":
        tmin = 1.2
    else:
        tmin = 0.5
    msk = (
        (arr['flags'] == 0)
        & (arr[f'{model}_s2n'] > 1000)
        & (arr[f'{model}_T_ratio'] > tmin)
    )
    return msk


def _meas_shear_data(res, model):
    msk = _shear_cuts(res['noshear'], model)
    g1 = np.mean(res['noshear'][f'{model}_g'][msk, 0])
    g2 = np.mean(res['noshear'][f'{model}_g'][msk, 1])

    msk = _shear_cuts(res['1p'], model)
    g1_1p = np.mean(res['1p'][f'{model}_g'][msk, 0])
    msk = _shear_cuts(res['1m'], model)
    g1_1m = np.mean(res['1m'][f'{model}_g'][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'], model)
    g2_2p = np.mean(res['2p'][f'{model}_g'][msk, 1])
    msk = _shear_cuts(res['2m'], model)
    g2_2m = np.mean(res['2m'][f'{model}_g'][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def get_shear(res, model="wmom"):
    return _meas_shear_data(res, model)
