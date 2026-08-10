"""
Most of the tests here are copied from ngmix/tests with adaptation to match
metacoadd implementation.
"""

import numpy as np
import pytest

import ngmix
from ngmix.tests._galsim_sims import _get_obs

from metacoadd.metacal import MetacalHandler


@pytest.mark.parametrize("mcal_class", ["gauss_psf", "fix_gauss_psf"])
def test_metacal_obs(mcal_class):
    """
    Test that the metacal handler returns a dictionary of metacal observations
    with the correct keys and that the images are different for each metacal
    type.

    Parameters
    ----------
    mcal_class : str
        The metacal class to use in ["gauss_psf", "fix_gauss_psf"].
    """
    rng = np.random.RandomState(1234)
    obs = _get_obs(rng, noise=0.005, set_noise_image=True)
    mcal_types = ngmix.metacal.METACAL_MINIMAL_TYPES

    mcal = MetacalHandler(
        rng=rng,
        mcal_class=mcal_class,
    )
    obs_dict = mcal.get_all(obs, mcal_types)

    assert len(obs_dict) == len(mcal_types)
    for mcal_type in mcal_types:
        assert mcal_type in obs_dict

        mcal_obs = obs_dict[mcal_type]
        assert mcal_obs.image.shape == obs.image.shape
        assert np.all(mcal_obs.image != obs.image)
        assert mcal_obs.psf.image.shape == obs.psf.image.shape
        assert np.all(mcal_obs.psf.image != obs.psf.image)

    assert np.all(obs_dict["1p"].image != obs_dict["1m"].image)
    assert np.all(obs_dict["2p"].image != obs_dict["2m"].image)
    assert np.all(obs_dict["noshear"].image != obs_dict["1p"].image)


@pytest.mark.parametrize("otype", ["obs", "obslist", "mbobs"])
def test_metacal_fixnoise_smoke(otype, set_noise_image=True):
    """
    Test that the metacal handler returns the correct kind of observation type
    depending on the input observation type.
    """
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005, set_noise_image=set_noise_image)

    if otype == "obslist":
        oobs = obs
        obs = ngmix.ObsList()
        obs.append(oobs)
        check_type = ngmix.ObsList
    elif otype == "mbobs":
        oobs = obs
        obslist = ngmix.ObsList()
        obslist.append(oobs)

        obs = ngmix.MultiBandObsList()
        obs.append(obslist)
        check_type = ngmix.MultiBandObsList
    else:
        check_type = ngmix.Observation

    mcal = MetacalHandler(
        rng=rng,
        mcal_class="gauss_psf",
    )
    obs_dict = mcal.get_all(obs, mcal_types=["noshear"])
    assert isinstance(obs_dict["noshear"], check_type)


@pytest.mark.parametrize("fixnoise", [True, False])
def test_metacal_fixnoise(fixnoise):
    """
    Test that the metacal handler apply the fixnoise correctly with the correct
    noise level and weights.
    """
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005, set_noise_image=True)

    mcal = MetacalHandler(
        rng=rng,
        mcal_class="gauss_psf",
        fixnoise=fixnoise,
    )
    obs_dict = mcal.get_all(obs, mcal_types=["noshear"])

    for _, mobs in obs_dict.items():
        assert mobs.image.shape == obs.image.shape
        assert np.all(mobs.image != obs.image)
        assert mobs.psf.image.shape == obs.psf.image.shape
        assert np.all(mobs.psf.image != obs.psf.image)
        if fixnoise:
            assert mobs.weight[0, 0] == obs.weight[0, 0] / 2
            assert mobs.pixels[0]["ierr"] == np.sqrt(obs.weight[0, 0] / 2)
        else:
            assert mobs.weight[0, 0] == obs.weight[0, 0]
            assert mobs.pixels[0]["ierr"] == np.sqrt(obs.weight[0, 0])


def _do_test_low_psf_s2n():
    rng = np.random.RandomState(seed=100)
    noise = 1000

    for _ in range(1000):
        obs = _get_obs(rng, noise=0.005, set_noise_image=True)

        with obs.psf.writeable():
            psf_im = obs.psf.image
            psf_wt = obs.psf.weight
            psf_im += rng.normal(scale=noise, size=psf_im.shape)
            psf_wt[:, :] = 1.0 / noise**2

        # ngmix.metacal.get_all_metacal(obs=obs, rng=rng, psf="fitgauss")
        mcal = MetacalHandler(
            rng=rng,
            mcal_class="gauss_psf",
        )
        mcal.get_all(obs, mcal_types=["noshear"])


def test_low_psf_s2n():
    """
    Test that the metacal handler raises an exception when the PSF S/N is too
    low.
    """
    with pytest.raises(ngmix.BootPSFFailure):
        _do_test_low_psf_s2n()


def test_metacal_noncontiguous():
    """
    Test that the metacal handler works with non-contiguous images.
    """
    rng = np.random.RandomState(seed=932)

    tobs = _get_obs(rng, noise=0.005, set_noise_image=True, n=48)

    timage = np.zeros(tobs.image.shape, dtype=">f8")
    timage[:, :] = tobs.image
    tsub_image = tobs.image[10 : 48 - 10, 10 : 48 - 10]
    tsub_noise = tobs.noise[10 : 48 - 10, 10 : 48 - 10]

    obs = ngmix.Observation(
        image=tsub_image,
        jacobian=tobs.jacobian,
        psf=tobs.psf,
        noise=tsub_noise,
    )

    mcal = MetacalHandler(
        rng=rng,
        mcal_class="gauss_psf",
    )
    _ = mcal.get_all(
        obs,
        mcal_types=["noshear"],
    )
