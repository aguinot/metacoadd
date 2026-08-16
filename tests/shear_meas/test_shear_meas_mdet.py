"""
This test has been taken from the metadetect repository and adapted for the
implementation here
The original test is here: https://github.com/esheldon/metadetect/blob/master/shear_meas_test/test_shear_meas.py
"""

import time
import copy

import numpy as np

import ngmix
import galsim

from esutil.pbar import PBar

import joblib
import pytest

from metacoadd.metadetect import MetaDetect


TEST_METADETECT_CONFIG = {
    "model": "wmom",
    "weight": {
        "fwhm": 1.2,
    },
    "metacal": {
        "psf": "fitgauss",
        "types": [
            "noshear",
            "1p",
            "1m",
            "2p",
            "2m",
        ],
    },
    "sx": {
        # In sky sigma.
        "detect_thresh": 0.8,
        # Minimum contrast parameter for deblending.
        "deblend_cont": 0.00001,
        # Minimum number of pixels above threshold.
        "minarea": 4,
        "filter_type": "conv",
        # 7x7 convolution mask of a Gaussian PSF with
        # FWHM = 3.0 pixels.
        "filter_kernel": [
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],
        ],
    },
    "meds": {
        "min_box_size": 32,
        "max_box_size": 32,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    # Check for an edge hit.
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}


def make_sim(
    *, seed, g1, g2, dim=251, buff=34, scale=0.25, dens=100, ngrid=7, snr=1e6
):
    """Make a simple simulation of a sheared galaxy convolved with a PSF."""
    rng = np.random.RandomState(
        seed=seed,
    )

    half_loc = (dim - buff * 2) * scale / 2

    if ngrid is None:
        area_arcmin2 = ((dim - buff * 2) * scale / 60) ** 2
        nobj = int(dens * area_arcmin2)
        x = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
        y = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
    else:
        half_ngrid = (ngrid - 1) / 2
        x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
        x = (x.ravel() - half_ngrid) / half_ngrid * half_loc
        y = (y.ravel() - half_ngrid) / half_ngrid * half_loc
        nobj = x.shape[0]

    cen = (dim - 1) / 2
    psf_dim = 53
    psf_cen = (psf_dim - 1) / 2

    psf = galsim.Gaussian(
        fwhm=0.9,
    )

    gals = []

    for ind in range(nobj):
        u, v = rng.uniform(low=-scale, high=scale, size=2)
        u += x[ind]
        v += y[ind]

        gals.append(galsim.Exponential(half_light_radius=0.5).shift(u, v))

    gals = galsim.Add(gals)
    gals = gals.shear(g1=g1, g2=g2)
    gals = galsim.Convolve([gals, psf])

    im = gals.drawImage(nx=dim, ny=dim, scale=scale).array

    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

    nse = (
        np.sqrt(
            np.sum(
                galsim.Convolve(
                    [psf, galsim.Exponential(half_light_radius=0.5)]
                )
                .drawImage(scale=0.25)
                .array
                ** 2
            )
        )
        / snr
    )

    im += rng.normal(
        size=im.shape,
        scale=nse,
    )

    # MetaCoadd uses the observation noise image for fixnoise.
    # This is independent of the noise added to the science image.
    noise = rng.normal(
        size=im.shape,
        scale=nse,
    )

    wgt = np.ones_like(im) / nse**2

    jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=cen,
        col=cen,
    )
    psf_jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=psf_cen,
        col=psf_cen,
    )

    obs = ngmix.Observation(
        image=im,
        weight=wgt,
        noise=noise,
        jacobian=jac,
        ormask=np.zeros_like(im, dtype=np.int32),
        bmask=np.zeros_like(im, dtype=np.int32),
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
    )

    obslist = ngmix.ObsList()
    obslist.append(obs)

    mbobs = ngmix.MultiBandObsList()
    mbobs.append(obslist)

    return mbobs


def _make_metadetect(config, rng):
    """Translate the upstream configuration to MetaCoadd."""
    model = config["model"]

    return MetaDetect(
        rng=rng,
        step=0.01,
        types=copy.deepcopy(
            config["metacal"]["types"],
        ),
        detect_type="relative",
        detect_thresh=config["sx"]["detect_thresh"],
        detect_minarea=config["sx"]["minarea"],
        detect_deblend_nthresh=32,
        detect_deblend_cont=config["sx"]["deblend_cont"],
        detect_kernel=copy.deepcopy(config["sx"]["filter_kernel"]),
        detect_filter_type=config["sx"]["filter_type"],
        coadd_type="average",
        coadd_multiband=True,
        coadd_fscale=[
            [1.0],
        ],
        models=[
            model,
        ],
        fwhms=[
            config["weight"]["fwhm"],
        ],
        symmetrizes=[
            True,
        ],
        stamp_size=config["meds"]["min_box_size"],
        do_uberseg=False,
        mcal_config=copy.deepcopy(
            config["metacal"],
        ),
    )


def _shear_cuts(arr, model):
    # tmin = 1.2 if model == "wmom" else 0.5
    # For some reason I had to lower the cut for wmom for the objects to pass
    # the cut.
    tmin = 0.5
    Tr = arr[f"{model}_T"] / arr[f"{model}_Tpsf"]

    msk = (
        (arr[f"{model}_flags"] == 0) & (arr[f"{model}_s2n"] > 10) & (Tr > tmin)
    )

    return msk


def _meas_shear_data(res, model):
    msk = _shear_cuts(res["noshear"], model)
    g1 = np.mean(res["noshear"][f"{model}_g1"][msk])
    g2 = np.mean(res["noshear"][f"{model}_g2"][msk])

    msk = _shear_cuts(res["1p"], model)
    g1_1p = np.mean(res["1p"][f"{model}_g1"][msk])

    msk = _shear_cuts(res["1m"], model)
    g1_1m = np.mean(res["1m"][f"{model}_g1"][msk])

    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res["2p"], model)
    g2_2p = np.mean(res["2p"][f"{model}_g2"][msk])

    msk = _shear_cuts(res["2m"], model)
    g2_2m = np.mean(res["2m"][f"{model}_g2"][msk])

    R22 = (g2_2p - g2_2m) / 0.02

    dtype = [
        ("g1", "f8"),
        ("g2", "f8"),
        ("R11", "f8"),
        ("R22", "f8"),
    ]

    return np.array([(g1, g2, R11, R22)], dtype=dtype)


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]

    rng = np.random.RandomState(seed=seed)

    stats = []

    for _ in range(nboot):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))

    return stats


def meas_m_c_cancel(pres, mres):
    """Compute the multiplicative and additive shear bias using shape noise
    cancellation.
    """
    x = np.mean(pres["g1"] - mres["g1"]) / 2
    y = np.mean(pres["R11"] + mres["R11"]) / 2
    m = x / y / 0.02 - 1

    x = np.mean(pres["g2"] + mres["g2"]) / 2
    y = np.mean(pres["R22"] + mres["R22"]) / 2
    c = x / y

    return m, c


def boostrap_m_c(pres, mres):
    """Compute the multiplicative and additive shear bias using shape noise
    cancellation and bootstrap resampling to estimate the uncertainty.
    """
    m, c = meas_m_c_cancel(pres, mres)

    bdata = _bootstrap_stat(pres, mres, meas_m_c_cancel, 14324, nboot=500)

    merr, cerr = np.std(bdata, axis=0)

    return m, merr, c, cerr


def run_sim(seed, mdet_seed, model, **kwargs):
    """Run a single simulation and measure the shear using MetaCoadd."""
    mbobs_p = make_sim(seed=seed, g1=0.02, g2=0.0, **kwargs)

    cfg = copy.deepcopy(TEST_METADETECT_CONFIG)
    cfg["model"] = model

    # Use a fresh RNG initialized with the same seed for the positive
    # and negative simulations, preserving paired-noise cancellation.
    mdet_p = _make_metadetect(cfg, np.random.RandomState(seed=mdet_seed))
    pres = mdet_p.go(mbobs_p)

    if pres is None:
        return None

    mbobs_m = make_sim(seed=seed, g1=-0.02, g2=0.0, **kwargs)
    mdet_m = _make_metadetect(cfg, np.random.RandomState(seed=mdet_seed))
    mres = mdet_m.go(mbobs_m)

    if mres is None:
        return None

    return _meas_shear_data(pres, model), _meas_shear_data(mres, model)


@pytest.mark.parametrize(
    "model, snr, ngrid, ntrial",
    [
        ("wmom", 1e6, 7, 64),
        ("pgauss", 1e6, 7, 64),
        ("gam", 1e6, 7, 64),
        # ("am", 1e6, 7, 64),
        ("gauss", 1e6, 7, 64),
        # This test takes about three hours in the original
        # metadetect GitHub Actions workflow.
        ("gam", 1e6, None, 9500),
    ],
)
def test_shear_meas_simple(model, snr, ngrid, ntrial):
    """Run a simple shear measurement test using the specified model and
    parameters.
    """
    nsub = max(
        ntrial // 128,
        8,
    )
    nitr = ntrial // nsub

    rng = np.random.RandomState(
        seed=116,
    )

    seeds = rng.randint(
        low=1,
        high=2**29,
        size=ntrial,
    )
    mdet_seeds = rng.randint(
        low=1,
        high=2**29,
        size=ntrial,
    )

    tm0 = time.time()

    print("")

    pres = []
    mres = []
    loc = 0

    with joblib.Parallel(
        n_jobs=-1,
        verbose=100,
        backend="loky",
    ) as parallel:
        for _ in PBar(
            range(nitr),
        ):
            jobs = [
                joblib.delayed(run_sim)(
                    seeds[loc + index],
                    mdet_seeds[loc + index],
                    model,
                    snr=snr,
                    ngrid=ngrid,
                )
                for index in range(nsub)
            ]

            print(
                "\n",
                end="",
                flush=True,
            )

            outputs = parallel(jobs)

            for output in outputs:
                if output is None:
                    continue

                pres.append(output[0])
                mres.append(output[1])

            loc += nsub

            m, merr, c, cerr = boostrap_m_c(
                np.concatenate(pres),
                np.concatenate(mres),
            )

            print(
                (
                    "\n"
                    f"nsims: {len(pres)}\n"
                    "m [1e-3, 3sigma]: "
                    f"{m / 1e-3:.2f} +/- {3 * merr / 1e-3:.2f}\n"
                    "c [1e-5, 3sigma]: "
                    f"{c / 1e-5:.2f} +/- {3 * cerr / 1e-5:.2f}\n"
                    "\n"
                ),
                flush=True,
            )

    total_time = time.time() - tm0

    print(
        "time per:",
        total_time / ntrial,
        flush=True,
    )

    pres = np.concatenate(pres)
    mres = np.concatenate(mres)

    m, merr, c, cerr = boostrap_m_c(
        pres,
        mres,
    )

    print(
        (
            f"m [1e-3, 3sigma]: {m / 1e-3:.2f} +/- {3 * merr / 1e-3:.2f}"
            f"\nc [1e-5, 3sigma]: {c / 1e-5:.2f} +/- {3 * cerr / 1e-5:.2f}"
        ),
        flush=True,
    )

    assert np.abs(m) < max(
        1e-3,
        3 * merr,
    )
    assert np.abs(c) < (3 * cerr)
