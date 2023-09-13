import warnings
from time import time

from astropy.utils.exceptions import AstropyWarning

from metacoadd.simu import get_shear, make_sim, run_metacoadd

warnings.simplefilter("ignore", category=AstropyWarning)

ts = time()
# Input params
# The center is linked to headers used for the test.
# Cannot be changed at the moment.
ra_center = 110.1844991  # Deg
dec_center = 52.8002126  # Deg
scale = 0.185768447408928  # Arcsec
cell_size = 1.0 / 60.0  # Arcmin

noise = 1e-5

params_obj = {
    "hlr": 0.7,
    "flux": 100,
    "g1": 0.01,
    "g2": 0.0,
}

params_single = {
    "psf_fwhm": 0.7,
    "psf_fwhm_std": 0.5,
    "psf_g1": 0.0,
    "psf_g2": 0.0,
    "noise": noise,
}


# Input data
input_headers_dir = "../data/pre_selection_3/"


# Make simu
explist, explist_psf, obj_dict = make_sim(
    input_headers_dir,
    ra_center,
    dec_center,
    scale,
    cell_size,
    params_obj,
    params_single,
    seed=1234,
)

# Run metacoadd
mc = run_metacoadd(
    explist, ra_center, dec_center, scale, cell_size, explist_psf=explist_psf
)
final_res = get_shear(mc.results, "wmom")

print(
    "m: {}\nc: {}".format(
        (final_res["g1"] / final_res["R11"] - params_obj["g1"])[0],
        (final_res["g2"] / final_res["R22"])[0],
    )
)
print("Took:", time() - ts, "s")
