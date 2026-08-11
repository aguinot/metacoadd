import galsim
import numpy as np
import ngmix

from metacoadd.metacal import MetacalHandler


def _get_obs(rng):
    """
    obs with noise image included
    """
    noise = 0.1
    psf_noise = 1.0e-6
    scale = 0.263

    psf_fwhm = 0.9
    gal_fwhm = 0.7

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    obj0 = galsim.Gaussian(fwhm=gal_fwhm)

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    cen = (np.array(im.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

    j = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    pj = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    wt = im * 0 + 1.0 / noise**2
    psf_wt = psf_im * 0 + 1.0 / psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=pj,
    )
    im += rng.normal(scale=noise, size=im.shape)
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    nim = rng.normal(scale=noise, size=im.shape)

    obs = ngmix.Observation(
        im,
        weight=wt,
        noise=nim,
        jacobian=j,
        psf=psf_obs,
    )

    return obs


def test_metacal_fixnoise_noise_image():
    """
    Test that the fixnoise works as expected.
    """
    rng = np.random.RandomState(seed=100)
    obs = _get_obs(rng)
    noise_obs = _get_obs(rng)

    with noise_obs.writeable():
        nim = obs.noise.copy()
        noise_obs.image[:, :] = np.rot90(nim, k=1)

    mcal_types = ["noshear", "1p", "1m"]
    mcal = MetacalHandler(
        rng=rng,
        use_noise_image=True,
        fixnoise=True,
        mcal_class="fix_gauss_psf",
        mcal_config={
            "fwhm_target": 0.9 * 1.02,
        },
    )
    mdict = mcal.get_all(obs, mcal_types=mcal_types)
    mcal_no_fixnoise = MetacalHandler(
        rng=rng,
        fixnoise=False,
        mcal_class="fix_gauss_psf",
        mcal_config={
            "fwhm_target": 0.9 * 1.02,
        },
    )
    mdict_no_fixnoise = mcal_no_fixnoise.get_all(obs, mcal_types=mcal_types)
    mcal_noise = MetacalHandler(
        rng=rng,
        fixnoise=False,
        mcal_class="fix_gauss_psf",
        mcal_config={
            "fwhm_target": 0.9 * 1.02,
        },
    )
    mdict_noise = mcal_noise.get_all(noise_obs, mcal_types=mcal_types)

    for key in mdict:
        im = mdict[key].image
        im_no_fixnoise = mdict_no_fixnoise[key].image
        noise_im = np.rot90(mdict_noise[key].image, k=3)

        assert np.all(im == im_no_fixnoise + noise_im)
        assert np.all(im != im_no_fixnoise)
