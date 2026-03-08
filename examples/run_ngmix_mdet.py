from copy import deepcopy

import numpy as np

import galsim

import ngmix

from roman_shear_sims.sim import make_sim
from roman_shear_sims.catalog import SimpleGalaxyCatalog
from roman_shear_sims.psf_makers import PSFMaker
from roman_shear_sims.constant import IMCOM_BLOCK_SIZE

import metadetect

from time import time


####
# Setup sims
####

seed = 4242
rng = np.random.RandomState(seed)

simu_type = "imcom"
simu_size = IMCOM_BLOCK_SIZE
bands = ["Y106"]  # , "J129", "H158"]
gal_type = "gauss"
psf_type = simu_type
layout_kind = "grid"
chromatic = False
spacing = 12.0
buff = 200
noise_sig = 1e-5
n_gal = None
gal_mag = 22
gal_hlr = 0.3
flux_range = [100, 1_00]

g1_in = 0.0
g2_in = 0.0

n_epochs = 1

exp_time = 107


galaxy_catalog = SimpleGalaxyCatalog(
    simu_size,
    seed,
    simu_type=simu_type,
    gal_type=gal_type,
    mag=gal_mag,
    hlr=gal_hlr,
    layout_kind=layout_kind,
    exp_time=exp_time,
    spacing=spacing,
    buffer=buff,
    n_gal=n_gal,
    chromatic=chromatic,
)
psf_maker = PSFMaker(
    psf_type=psf_type,
    chromatic=chromatic,
)
print("Setup sims done")


####
# Run sims
####

simu_dict = make_sim(
    rng,
    galaxy_catalog,
    psf_maker,
    simu_type=simu_type,
    n_epochs=n_epochs,
    exp_time=exp_time,
    cell_size_pix=simu_size,
    bands=bands,
    g1=g1_in,
    g2=g2_in,
    chromatic=chromatic,
    simple_noise=True,
    noise_sigma=noise_sig,
    draw_method="fft",
    verbose=False,
)
print("Run sims done")


####
# Setup data
####

mbobs = ngmix.MultiBandObsList()
obslist = ngmix.ObsList()
for band in bands:
    for i in range(n_epochs):
        wcs = simu_dict[band][i]["wcs"]
        h = wcs.header
        g_jacob = wcs.jacobian(
            image_pos=galsim.PositionD(h["CRPIX1"], h["CRPIX2"])
        )

        img = simu_dict[band][i]["sci"][f"shear_{g1_in}_{g2_in}"]
        img_cen = (np.array(img.shape) - 1) / 2.0
        img_jacob = ngmix.Jacobian(
            row=img_cen[1],
            col=img_cen[0],
            wcs=g_jacob,
        )

        psf_img = simu_dict[band][i]["psf"]
        psf_cen = (np.array(psf_img.shape) - 1) / 2.0
        psf_jacob = ngmix.Jacobian(
            row=psf_cen[1],
            col=psf_cen[0],
            wcs=g_jacob,
        )

        psf_obs = ngmix.Observation(
            image=psf_img,
            jacobian=psf_jacob,
        )

        obs = ngmix.Observation(
            image=img,
            weight=simu_dict["Y106"][0]["weight"],
            noise=simu_dict[band][i]["noise"],
            psf=psf_obs,
            jacobian=img_jacob,
            ormask=np.zeros(img.shape, dtype=np.int32),
            bmask=np.zeros(img.shape, dtype=np.int32),
        )

        obslist.append(obs)
    mbobs.append(obslist)

METADETECT_CONFIG = {
    # Shape measurement method
    # wmom: weighted moments
    "model": "wmom",
    # Size of the weight function for the moments
    "weight": {
        "fwhm": 1.2,  # arcsec
    },
    # Metacal settings
    "metacal": {
        "psf": "fitgauss",
        # Kind of shear applied to the image
        "types": ["noshear", "1p", "1m", "2p", "2m"],
        "use_noise_image": True,
        "fixnoise": True,
    },
    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 1500,
        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        "deblend_cont": 0.005,
        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 5,
        "filter_type": "conv",
        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
        ],
    },
    # This is for the cutout at each detection
    "meds": {
        "min_box_size": 31,
        "max_box_size": 31,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    # check for an edge hit
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}
print("Setup data done")


####
# Run mdet
####
if __name__ == "__main__":
    ts = time()
    rng = np.random.RandomState(42)
    res = metadetect.do_metadetect(
        deepcopy(METADETECT_CONFIG),
        mbobs=mbobs,
        rng=rng,
    )
    print("Run mdet done, time taken: ", time() - ts)
