import galsim
from astropy.io.fits import Header
import ngmix
import metadetect as mdet
import metacoadd as mtc

import numpy as np
from math import ceil

import os


TEST_METADETECT_CONFIG = {
    "model": "wmom",

    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': False,
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
    'psf': {
        'model': 'gauss',

        'ntry': 2,

        'lm_pars': {
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        }
    },

    # check for an edge hit
    'bmask_flags': 2**30,

    'nodet_flags': 2**0,
}


def make_sim(
    headers_dir,
    coadd_ra,
    coadd_dec,
    coadd_scale,
    coadd_size,
    params_obj,
    params_single,
    seed,
    grid=True,
    sep_size=50,
    get_psf=True,
    get_obj_dict=True,
):
    """Make simu

    Build a set of single exposures to test metacoadd.

    Args:
        headers_dir (str): Path to the directory with the header from which the
            WCS arederived.
        coadd_ra (float): RA position of the coadd center.
        coadd_dec (float): DEC position of the coadd center.
        coadd_scale (float): Pixel scale to use for the coadd.
        coadd_size (float): Size of the coadd (in arcmin).
        params_obj (dict): Dictionnary with the information to build the object
            catalog.
        params_single (dict): Dictionnary with the general informations to
            build the single exposures.
        seed (int): RNG seed.
        grid (bool, optional): If `True`, will place objects on a grid.
            Randomly distributed otherwise. Defaults to True.
        sep_size (int, optional): Separetion to use for between objects. Only
            used for the grid. Defaults to 50.
        get_psf (bool, optional): If `True`, will return a `metacoadd.Exposure`
            instance with the PSF information. Defaults to True.
        get_obj_dict (bool, optional): If `True`, will return a the object
            catalog as a `dict`. Defaults to True.

    Returns:
        metacoadd.ExpList or tuple: Returns the created exposure list or a
            tuple with optionaly the PSF and/or the object catalog.
    """

    np_rng = np.random.default_rng(seed=seed)

    headers_list = os.listdir(headers_dir)
    coadd_size_image = ceil(coadd_size*3600/coadd_scale)

    # Create object catalog
    # print("Build object catalog..")
    obj_dict = make_obj_dict(
        coadd_size_image,
        coadd_ra,
        coadd_dec,
        coadd_scale,
        params_obj,
        grid=grid,
        sep_size=sep_size,
    )

    # Create exposures
    # print("Build all exposures..")
    explist = mtc.ExpList()
    if get_psf:
        explist_psf = mtc.ExpList()
    for header_path in headers_list:

        params_single_tmp = params_single.copy()
        params_single_tmp['seed'] = np_rng.integers(low=1, high=2**30)

        exp_dict = make_single_exp(
            params_single_tmp,
            headers_dir + header_path,
            obj_dict,
        )
        if exp_dict is None:
            # print(f'{header_path} skipped.')
            continue

        coadd_center_on_exp = exp_dict['wcs'].toImage(
            galsim.CelestialCoord(
                    ra=coadd_ra*galsim.degrees,
                    dec=coadd_dec*galsim.degrees,
                )
        )
        psf_img_local = exp_dict['psf'].drawImage(
            center=coadd_center_on_exp,
            nx=51,
            ny=51,
            wcs=exp_dict['wcs'],
        )

        psf_weight = np.ones_like(psf_img_local.array) / 1e-5**2.
        psf_weight_galsim = galsim.Image(
            psf_weight,
            bounds=psf_img_local.bounds,
        )

        exp = mtc.Exposure(
            image=exp_dict['image'],
            noise=exp_dict['noise'],
            weight=exp_dict['weight'],
            wcs=exp_dict['wcs'],
        )
        explist.append(exp)

        if get_psf:
            exp_psf = mtc.Exposure(
                image=psf_img_local,
                weight=psf_weight_galsim,
                wcs=exp_dict['wcs'],
            )
            explist_psf.append(exp_psf)

    output = (explist, )
    if get_psf:
        output += (explist_psf, )
    if get_obj_dict:
        output += (obj_dict, )

    return output


def make_obj_dict(
    img_size,
    coadd_ra,
    coadd_dec,
    coadd_scale,
    params_obj,
    grid=True,
    sep_size=50,
):
    """Make object dict

    Create an object catalog.
    NOTE: At the moment the objects are placed on a grid.

    Args:
        img_size (int): Size of the coadd in pixels
        coadd_ra (float): RA position of the coadd center.
        coadd_dec (float): DEC position of the coadd center.
        coadd_scale (float): Pixel scale to use for the coadd.
        params_obj (dict): Dictionnary with the information to build the object
            catalog.
        grid (bool, optional): If `True`, will place objects on a grid.
            Randomly distributed otherwise. Defaults to True.
        sep_size (int, optional): Separetion to use for between objects. Only
            used for the grid. Defaults to 50.

    Returns:
        dict : Object catalog.
    """

    coadd_cen = (np.array([img_size]*2)-1)/2
    gal_coadd_cen = galsim.PositionD(
        x=coadd_cen[0],
        y=coadd_cen[1]
    )

    coadd_world_cen = galsim.CelestialCoord(
        ra=coadd_ra*galsim.degrees,
        dec=coadd_dec*galsim.degrees,
    )

    coadd_wcs = make_coadd_wcs(
        scale=coadd_scale,
        image_origin=gal_coadd_cen,
        world_origin=coadd_world_cen,
    )

    if grid:
        if 'sep_size' in list(params_obj.keys()):
            sep_size = params_obj['sep_size']

        n_obj_x = int(img_size / sep_size)
        init_step_x = (img_size - n_obj_x*sep_size) // 2

        n_obj_y = int(img_size / sep_size)
        init_step_y = (img_size - n_obj_y*sep_size) // 2

        img_pos_x = np.array(
            [
                init_step_x + sep_size/2 + i*sep_size for i in range(n_obj_x)
            ]*(n_obj_y)
        )

        img_pos_y = np.array(
            [
                [init_step_y + sep_size/2 + i*sep_size]*(n_obj_x)
                for i in range(n_obj_y)
            ]
        ).ravel()

        n_obj = n_obj_x * n_obj_y

    obj_dict = {
        'coadd_pos': np.array([]),
        'world_pos': np.array([]),
        'gal_gs': np.array([]),
    }
    for i in range(n_obj):
        obj_dict['coadd_pos'] = np.append(
            obj_dict['coadd_pos'],
            galsim.PositionD(img_pos_x[i], img_pos_y[i])
        )
        obj_dict['world_pos'] = np.append(
            obj_dict['world_pos'],
            coadd_wcs.toWorld(obj_dict['coadd_pos'][-1]),
        )

        gal = galsim.Gaussian(
            half_light_radius=params_obj['hlr']
        ).withFlux(params_obj['flux'])
        gal = gal.shear(
            g1=params_obj['g1'],
            g2=params_obj['g2']
        )

        obj_dict['gal_gs'] = np.append(obj_dict['gal_gs'], gal)

    return obj_dict


def make_coadd_wcs(*, scale, image_origin, world_origin):
    """
    make and return a wcs object
    Parameters
    ----------
    scale: float
        Pixel scale
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """

    mat = np.array(
        [[scale, 0.0],
         [0.0, scale]],
    )

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_single_exp(
    params_single,
    header_path,
    obj_dict,
):
    """Make single exposure

    Create a single exposure image.

    Args:
        params_single (dict): Dictionnary with the general informations to
            build the single exposures.
        headers_dir (str): Path to the directory with the header from which the
            WCS arederived.
        obj_dict (dict) : Object catalog.

    Returns:
        dict: `dict` containing the image, weight, noise, wcs and psf.
    """

    seed = params_single['seed']
    np_rng = np.random.default_rng(seed=seed)

    # Get true WCS
    img_header = Header.fromfile(header_path)
    wcs = galsim.AstropyWCS(header=img_header)
    nx = img_header['NAXIS1']
    ny = img_header['NAXIS2']

    # Make image
    image = galsim.Image(
        nx,
        ny,
        wcs=wcs,
    )
    weight_image = galsim.Image(
        nx,
        ny,
        wcs=wcs,
    )
    np_noise = np_rng.normal(size=(nx, ny)).T * params_single['noise']
    noise_image = galsim.Image(
        np_noise,
        wcs=wcs,
    )

    # Make PSF
    psf_fwhm = params_single['psf_fwhm']
    if params_single['psf_fwhm_std'] != 0:
        psf_fwhm = np_rng.normal()*params_single['psf_fwhm_std']
    psf = galsim.Gaussian(fwhm=psf_fwhm).withFlux(1)
    psf = psf.shear(g1=params_single['psf_g1'], g2=params_single['psf_g2'])

    # Add object to the image
    obj_in = draw_obj(image, obj_dict, psf)
    if obj_in == 0:
        return None

    # Add noise to final image
    final_img = image + noise_image

    # Weight image
    weight_image.fill(1/params_single['noise']**2.)

    exp_dict = {
        'image': final_img.array,
        'gal_img': final_img,
        'weight': weight_image.array,
        'noise': noise_image.array,
        'wcs': wcs,
        'psf': psf,
    }

    return exp_dict


def draw_obj(image, obj_dict, psf):
    """Draw obj

    Draw all objects on a `galsim.Image`.

    Args:
        image (galsim.Image): Image on which we draw the objects.
        obj_dict (dict) : Object catalog.
        psf (galsim.GSObject): `galsim.GSObject` describing the PSF.

    Returns:
        int: Number of object that has been drawn on the image.
    """

    wcs = image.wcs

    obj_in = 0
    for world_pos, gal_gs in zip(obj_dict['world_pos'], obj_dict['gal_gs']):
        image_pos = wcs.toImage(world_pos)
        local_wcs = wcs.local(world_pos=world_pos)

        final_gal = galsim.Convolve((gal_gs, psf))
        stamp = final_gal.drawImage(
            center=image_pos,
            wcs=local_wcs,
        )

        b = image.bounds & stamp.bounds
        if b.isDefined():
            image[b] += stamp[b]
            obj_in += 1

    return obj_in


def make_perfect_coadd(
    obj_dict,
    coadd_ra,
    coadd_dec,
    coadd_scale,
    coadd_size,
    params_single,
    seed=1234,
):
    """Make perfect coadd

    Make a "perfect" coadd image.

    Args:
        obj_dict (dict) : Object catalog.
        coadd_ra (float): RA position of the coadd center.
        coadd_dec (float): DEC position of the coadd center.
        coadd_scale (float): Pixel scale to use for the coadd.
        coadd_size (float): Size of the coadd (in arcmin).
        params_single (dict): Dictionnary with the general informations to
            build the single exposures.
        seed (int): RNG seed. Default to 1234.

    Returns:
        numpy.ndarray: Array with the perfect coadd.
    """

    coadd_size_image = ceil(coadd_size*3600/coadd_scale)

    coadd_cen = (np.array([coadd_size_image]*2)-1)/2
    gal_coadd_cen = galsim.PositionD(
        x=coadd_cen[0],
        y=coadd_cen[1]
    )

    coadd_world_cen = galsim.CelestialCoord(
        ra=coadd_ra*galsim.degrees,
        dec=coadd_dec*galsim.degrees,
    )

    coadd_wcs = make_coadd_wcs(
        scale=coadd_scale,
        image_origin=gal_coadd_cen,
        world_origin=coadd_world_cen,
    )

    coadd_image = galsim.Image(
        coadd_size_image,
        coadd_size_image,
        wcs=coadd_wcs,
    )

    psf_fwhm = params_single['psf_fwhm']
    coadd_psf = galsim.Gaussian(fwhm=psf_fwhm).withFlux(1)
    coadd_psf = coadd_psf.shear(
        g1=params_single['psf_g1'],
        g2=params_single['psf_g2'],
    )

    draw_obj(coadd_image, obj_dict, coadd_psf)

    np_rng = np.random.default_rng(seed)
    np_noise = np_rng.normal(
        size=(coadd_size_image, coadd_size_image)
    ).T * 1e-5
    noise_image = galsim.Image(
        np_noise,
        wcs=coadd_wcs,
    )

    final_image = coadd_image + noise_image

    return final_image.array


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

    simplecoadd = mtc.SimpleCoadd(coaddimage)
    simplecoadd.go()
    output = (simplecoadd, )

    # Process PSFs
    if explist_psf is not None:
        coaddimage_psf = mtc.CoaddImage(
            explist=explist_psf,
            world_coadd_center=galsim.CelestialCoord(
                ra=coadd_ra*galsim.degrees,
                dec=coadd_dec*galsim.degrees,
            ),
            scale=coadd_scale,
            image_coadd_size=51,
            resize_exposure=False,
        )
        coaddimage_psf.get_all_interp_images()

        simplecoadd_psf = mtc.SimpleCoadd(coaddimage_psf)
        simplecoadd_psf.go()

        output += (simplecoadd_psf, )

    return output


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


def _shear_cuts(arr, model):
    if model == "wmom":
        tmin = 1.2
    else:
        tmin = 0.5
    msk = (
        (arr['flags'] == 0)
        & (arr[f'{model}_s2n'] > 10)
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
