from time import time

import numpy as np

from galsim_sim import _get_obs
from metacoadd.metacal import MetacalFixGaussPSF


def test_stepk_maxk():
    """
    Test that feeding the stepk and maxk lead to the same result as recomputing
    them.
    """

    rng = np.random.RandomState(1234)
    obs = _get_obs(rng, psf_fwhm=0.2, set_noise_image=True, n=1024)
    noise_obs = _get_obs(rng, psf_fwhm=0.2, set_noise_image=True, n=1024)

    with noise_obs.writeable():
        nim = obs.noise.copy()
        noise_obs.image[:, :] = np.rot90(nim, k=1)

    mcal = MetacalFixGaussPSF(
        obs,
        step=0.01,
        fwhm_target=0.2 * 1.02,
        rng=rng,
    )
    mcal_noise = MetacalFixGaussPSF(
        noise_obs,
        step=0.01,
        fwhm_target=0.2 * 1.02,
        rng=rng,
    )

    # Re-use stepk and maxk
    ts = time()
    stepk, maxk = mcal._set_data(obs)
    mcal_obs = mcal.get_obs_galshear(mcal_type="noshear")
    _, _ = mcal_noise._set_data(
        noise_obs,
        stepk=stepk,
        maxk=maxk,
    )
    mcal_obs_noise = mcal_noise.get_obs_galshear(mcal_type="noshear")
    final_image_sk_mk = mcal_obs.image + np.rot90(mcal_obs_noise.image, k=3)
    time_sk_mk = time() - ts

    # Clear data
    mcal._clear_data()
    mcal_noise._clear_data()

    # Recompute stepk and maxk
    ts = time()
    _, _ = mcal._set_data(obs)
    mcal_obs = mcal.get_obs_galshear(mcal_type="noshear")
    _, _ = mcal_noise._set_data(noise_obs)
    mcal_obs_noise = mcal_noise.get_obs_galshear(mcal_type="noshear")
    final_image_no_sk_mk = mcal_obs.image + np.rot90(mcal_obs_noise.image, k=3)
    time_no_sk_mk = time() - ts

    assert np.allclose(final_image_sk_mk, final_image_no_sk_mk)
    # Saving stepk and maxk should be faster than recomputing them but the
    # difference is only noticeable for large images.
    assert time_sk_mk < time_no_sk_mk
