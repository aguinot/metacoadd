import galsim
from astropy.io import fits

import numpy as np
from math import ceil

from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt

from metacoadd.exposure import Exposure, ExpList, CoaddImage
from metacoadd.metacoadd import SimpleCoadd

# Defaults

nx = 512
ny = 1024
ra = 220.0
dec = 55
pixel_scale = 0.187

np_rng = np.random.default_rng(seed=1234)

noise = 1e-3

nx = 2112
ny = 4644

h = fits.getheader(
    "/Users/aguinot/Documents/pipeline/simu_MICE/output/images/simu_image-2236611.fits",
    1,
)
wcs = galsim.AstropyWCS(header=h)

pos = wcs.toWorld(galsim.PositionD(x=nx / 2, y=ny / 2))
ra = pos.ra.deg
dec = pos.dec.deg
pixel_scale = np.sqrt(
    wcs.pixelArea(image_pos=galsim.PositionD(x=nx / 2, y=ny / 2))
)

params_coadd = {
    "field_center_ra": ra - 0.05,
    "field_center_dec": dec - 0.12,
    "field_size": 1.0 / 60,
    "pixel_scale": 0.187,
    "seed": 1234,
}

params_single = {
    "n_obj": 25,  # ignored if 'grid' is used
    "field_center_ra": ra,
    "field_center_dec": dec,
    "rotate": 15,
    "nx": nx,
    "ny": ny,
    "noise": noise,
    "pixel_scale": pixel_scale,
    "seed": 1233,
}

params_single2 = {
    "n_obj": 25,  # ignored if 'grid' is used
    "field_center_ra": ra,
    "field_center_dec": dec,
    "rotate": 0,
    "nx": nx,
    "ny": ny,
    "noise": noise,
    "pixel_scale": pixel_scale,
    "seed": 5678,
}

params_single3 = {
    "n_obj": 25,  # ignored if 'grid' is used
    "field_center_ra": ra,
    "field_center_dec": dec,
    "rotate": 0,
    "nx": nx,
    "ny": ny,
    "noise": noise,
    "pixel_scale": pixel_scale,
    "seed": 9101,
}

params_single_coadd = {
    "n_obj": 25,  # ignored if 'grid' is used
    "field_center_ra": ra - 0.05,
    "field_center_dec": dec - 0.12,
    "rotate": 0,
    "nx": ceil(params_coadd["field_size"] * 3600 / params_coadd["pixel_scale"]),
    "ny": ceil(params_coadd["field_size"] * 3600 / params_coadd["pixel_scale"]),
    "noise": noise,
    "pixel_scale": 0.187,
    "seed": 1234,
}

params_obj = {
    "n_obj": 25,  # ignored if 'grid' is used
    "hlr": 0.7,
    "flux": 100,
    "g1": 0.01,
    "g2": 0.0,
    "psf_fwhm": 0.7,
    "psf_fwhm_std": 0.01,
    "psf_g1": 0.0,
    "psf_g2": 0.0,
    "seed": 1234,
}

# Basic functions


def make_wcs1(
    *, scale, image_origin, world_origin, theta=None, scale_wcs=False, test=True
):
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
    theta: float, optional
        Rotation angle in radians
    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """
    if scale_wcs:
        return galsim.PixelScale(scale=scale)

    if not test:
        mat = np.array(
            [[scale, 0.0], [0.0, scale]],
        )
    else:
        mat = np.array([[0.18344354, 0.00748749], [0.00748749, 0.19093103]])

    if theta is not None:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        rot = np.array(
            [[costheta, -sintheta], [sintheta, costheta]],
        )
        mat = np.dot(mat, rot)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0],
            mat[0, 1],
            mat[1, 0],
            mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_wcs(
    *, scale, image_origin, world_origin, theta=None, scale_wcs=False, test=True
):
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
    theta: float, optional
        Rotation angle in radians
    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """
    h = fits.getheader(
        "/Users/aguinot/Documents/pipeline/simu_MICE/output/images/simu_image-2236611.fits",
        1,
    )
    wcs = galsim.AstropyWCS(header=h)

    return wcs


def make_gal(hlr, flux, g1, g2, psf_fwhm, psf_g1, psf_g2):
    """

    return
    gsobject
    """

    gal = galsim.Gaussian(half_light_radius=hlr).withFlux(flux)
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    psf = psf.shear(g1=psf_g1, g2=psf_g2)

    obj = galsim.Convolve((gal, psf))

    return obj, psf


def draw_obj(image, obj_list, noise):
    """ """

    wcs = image.wcs

    for obj in obj_list:
        image_pos = wcs.toImage(obj["world_pos"])

        local_wcs = wcs.local(world_pos=obj["world_pos"])
        stamp = obj["gal_image"].drawImage(center=image_pos, wcs=local_wcs)

        b = image.bounds & stamp.bounds
        if b.isDefined():
            image[b] += stamp[b]


def make_obj_list(
    params_coadd, params_obj, var_psf=False, grid=True, sep_size=25, test=False
):
    """ """

    img_size = ceil(
        params_coadd["field_size"] * 3600 / params_coadd["pixel_scale"]
    )

    coadd_cen = (np.array([img_size] * 2) - 1) / 2
    gal_coadd_cen = galsim.PositionD(x=coadd_cen[0], y=coadd_cen[1])

    coadd_world_cen = galsim.CelestialCoord(
        ra=params_coadd["field_center_ra"] * galsim.degrees,
        dec=params_coadd["field_center_dec"] * galsim.degrees,
    )

    coadd_wcs = make_wcs1(
        scale=params_coadd["pixel_scale"],
        theta=None,
        image_origin=gal_coadd_cen,
        world_origin=coadd_world_cen,
        test=False,
    )

    if grid:
        if "sep_size" in list(params_obj.keys()):
            sep_size = params_obj["sep_size"]

        n_obj_x = int(img_size / sep_size)
        init_step_x = (img_size - n_obj_x * sep_size) // 2

        n_obj_y = int(img_size / sep_size)
        init_step_y = (img_size - n_obj_y * sep_size) // 2

        img_pos_x = np.array(
            [init_step_x + sep_size / 2 + i * sep_size for i in range(n_obj_x)]
            * (n_obj_y)
        )
        img_pos_y = np.array(
            [
                [init_step_y + sep_size / 2 + i * sep_size] * (n_obj_x)
                for i in range(n_obj_y)
            ]
        ).ravel()

        n_obj = n_obj_x * n_obj_y

    obj_list = []
    for i in tqdm(range(n_obj), total=n_obj):
        obj_dict = {}
        obj_dict["coadd_pos"] = galsim.PositionD(img_pos_x[i], img_pos_y[i])
        obj_dict["world_pos"] = coadd_wcs.toWorld(obj_dict["coadd_pos"])
        obj, psf = make_gal(
            params_obj["hlr"],
            params_obj["flux"],
            params_obj["g1"],
            params_obj["g2"],
            params_obj["psf_fwhm"],
            params_obj["psf_g1"],
            params_obj["psf_g2"],
        )
        obj_dict["gal_image"] = obj
        obj_dict["psf_image"] = psf

        obj_list.append(obj_dict)

    return obj_list


def make_single_exp(
    params_single,
    params_coadd,
    params_obj,
    scale_wcs=False,
    test=True,
    single=True,
):
    """ """

    seed = params_single["seed"]
    np_rng = np.random.default_rng(seed=seed)

    img_cen = [(params_single["nx"] - 1) / 2, (params_single["ny"] - 1) / 2]
    gal_img_cen = galsim.PositionD(x=img_cen[0], y=img_cen[1])
    world_cen = galsim.CelestialCoord(
        ra=params_single["field_center_ra"] * galsim.degrees,
        dec=params_single["field_center_dec"] * galsim.degrees,
    )

    if single:
        wcs = make_wcs(
            scale=params_single["pixel_scale"],
            theta=params_single["rotate"] * np.pi / 180,
            image_origin=gal_img_cen,
            world_origin=world_cen,
            scale_wcs=scale_wcs,
            test=test,
        )
    else:
        wcs = make_wcs1(
            scale=params_single["pixel_scale"],
            theta=params_single["rotate"] * np.pi / 180,
            image_origin=gal_img_cen,
            world_origin=world_cen,
            scale_wcs=scale_wcs,
            test=test,
        )

    image = galsim.Image(params_single["nx"], params_single["ny"], wcs=wcs)
    weight_image = galsim.Image(
        params_single["nx"], params_single["ny"], wcs=wcs
    )

    obj_list = make_obj_list(params_coadd, params_obj, test=test)

    draw_obj(image, obj_list, params_single["noise"])

    final_img = image.array

    noise = np_rng.normal(size=final_img.shape) * params_single["noise"]
    final_img += noise

    weight_image.fill(1 / params_single["noise"] ** 2.0)

    exp_dict = {
        "image": final_img,
        "gal_img": image,
        "weight": weight_image.array,
        "wcs": wcs,
        "obj_list": obj_list,
    }

    return exp_dict


exp_dict = make_single_exp(params_single, params_coadd, params_obj)
exp_dict2 = make_single_exp(params_single2, params_coadd, params_obj)
exp_dict3 = make_single_exp(params_single3, params_coadd, params_obj)

perfect_coadd = make_single_exp(
    params_single_coadd, params_coadd, params_obj, test=False, single=False
)

# METACOADD
ts = time()
exp_1 = Exposure(
    image=exp_dict["image"], wcs=exp_dict["wcs"], weight=exp_dict["weight"]
)
exp_2 = Exposure(
    image=exp_dict2["image"], wcs=exp_dict2["wcs"], weight=exp_dict2["weight"]
)
exp_3 = Exposure(
    image=exp_dict3["image"], wcs=exp_dict3["wcs"], weight=exp_dict3["weight"]
)
explist = ExpList()
explist.append(exp_1)
explist.append(exp_2)
explist.append(exp_3)

coaddimage = CoaddImage(
    explist=explist,
    world_coadd_center=galsim.CelestialCoord(
        ra=params_coadd["field_center_ra"] * galsim.degrees,
        dec=params_coadd["field_center_dec"] * galsim.degrees,
    ),
    scale=0.187,
    image_coadd_size=ceil(
        params_coadd["field_size"] * 3600 / params_coadd["pixel_scale"]
    ),
)

coaddimage.get_all_interp_images()

simplecoadd = SimpleCoadd(coaddimage)

simplecoadd.go()
print(f"Total process: {time()-ts}s")

plt.figure(figsize=(7, 7))
plt.imshow(simplecoadd.coaddimage.image.array)
for obj in perfect_coadd["obj_list"]:
    plt.plot(obj["coadd_pos"].x + 0.5, obj["coadd_pos"].y + 0.5, "k+")
plt.colorbar()
plt.title("Coadd image")

plt.figure(figsize=(7, 7))
plt.imshow(simplecoadd.coaddimage.weight.array)
for obj in perfect_coadd["obj_list"]:
    plt.plot(obj["coadd_pos"].x, obj["coadd_pos"].y, "k+")
plt.colorbar()
plt.title("Coadd weight")

plt.show()
