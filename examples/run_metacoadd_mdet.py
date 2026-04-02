from copy import deepcopy

import numpy as np

import galsim

import ngmix

from roman_shear_sims.sim import make_sim
from roman_shear_sims.catalog import SimpleGalaxyCatalog
from roman_shear_sims.psf_makers import PSFMaker
from roman_shear_sims.constant import IMCOM_BLOCK_SIZE

from metacoadd.metadetect import MetaDetect
# from metacoadd.moments.galsim_admom import GAdmomFitter

from time import time


####
# Setup sims
####

seed = 4242
rng = np.random.RandomState(seed)

simu_type = "imcom"
simu_size = IMCOM_BLOCK_SIZE
bands = ["Y106", "J129", "H158"]
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
for band in bands:
    obslist = ngmix.ObsList()
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
            image=deepcopy(psf_img),
            jacobian=psf_jacob,
        )

        obs = ngmix.Observation(
            image=deepcopy(img),
            weight=deepcopy(simu_dict["Y106"][0]["weight"]),
            noise=deepcopy(simu_dict[band][i]["noise"]),
            psf=psf_obs,
            jacobian=img_jacob,
            ormask=np.zeros(img.shape, dtype=np.int32),
            bmask=np.zeros(img.shape, dtype=np.int32),
        )

        obslist.append(obs)
    mbobs.append(obslist)

print("Setup data done")


####
# Run mdet
####
if __name__ == "__main__":
    ts = time()

    gal_fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)
    gal_runner = ngmix.runners.Runner(fitter=gal_fitter)

    # psf_fitter = GAdmomFitter(guess_fwhm=0.6)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=0.6)
    psf_runner = ngmix.runners.Runner(fitter=psf_fitter)

    rng = np.random.RandomState(42)
    mdet = MetaDetect(
        rng=rng,
        psf_runner=psf_runner,
        gal_runners={
            "wmom": gal_runner,
        },
    )
    final_cat = mdet.go(mbobs)
    print("Run mdet done, time taken: ", time() - ts)
