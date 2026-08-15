"""Scientific integration tests for metacoadd.metadetect.

The simulated observations and the overall test structure are based on:

https://github.com/esheldon/metadetect/blob/master/metadetect/tests/
test_metadetect.py
"""

import copy

import numpy as np
import pytest

from metadetect import metadetect as ngmix_metadetect
from metadetect import fitting as ngmix_fitting

from metadetect.tests.sim import Sim, make_mbobs_sim

from metacoadd.metadetect import MetaDetect, do_metadetect
from metacoadd.detect import get_cutout
from metacoadd.detect import get_stamp_mbobs


SHEAR_TYPES = [
    "noshear",
    "1p",
    "1m",
    "2p",
    "2m",
]

SUPPORTED_MODELS = [
    "wmom",
    "pgauss",
    # "am",
    "gauss",
]

FILTER_KERNEL = [
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
]


MDET_CONFIG = {
    "model": "wmom",
    "weight": {
        "fwhm": 1.2,
    },
    "metacal": {
        "psf": "fitgauss",
        "types": SHEAR_TYPES,
    },
    "sx": {
        "detect_thresh": 0.8,
        "deblend_cont": 1.0e-5,
        "minarea": 4,
        "filter_type": "conv",
        "filter_kernel": FILTER_KERNEL,
    },
    "meds": {
        "min_box_size": 32,
        "max_box_size": 256,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}


def _get_fitter_config(model):
    fitter = {
        "model": model,
        "symmetrize": True,
    }

    if model in {"wmom", "pgauss"}:
        fitter["weight"] = {
            "fwhm": 1.2,
        }

    return fitter


def _get_metacoadd_config(
    *,
    nband=3,
    models=("wmom",),
    weight_type=None,
):
    """Translate the upstream config to the metacoadd architecture."""
    config = {
        "metacal": {
            "step": 0.01,
            "psf": "fitgauss",
            "types": SHEAR_TYPES,
            "use_noise_image": True,
            "fixnoise": True,
            "has_pixel": False,
        },
        "sx": {
            "detect_type": "relative",
            "detect_thresh": MDET_CONFIG["sx"]["detect_thresh"],
            "detect_minarea": MDET_CONFIG["sx"]["minarea"],
            "deblend_nthresh": 32,
            "deblend_cont": MDET_CONFIG["sx"]["deblend_cont"],
            "filter_type": MDET_CONFIG["sx"]["filter_type"],
            "filter_kernel": copy.deepcopy(MDET_CONFIG["sx"]["filter_kernel"]),
        },
        "meds": {
            "min_box_size": MDET_CONFIG["meds"]["min_box_size"],
            "max_box_size": MDET_CONFIG["meds"]["max_box_size"],
        },
        "coadd": {
            "type": "average",
            "fscale": [[1.0] for _ in range(nband)],
        },
        "fitters": [_get_fitter_config(model) for model in models],
    }

    if weight_type is not None:
        config["meds"]["weight_type"] = weight_type

    return config


def _make_driver(config, rng):
    """Construct MetaDetect from a standard configuration."""
    fitters = config["fitters"]

    return MetaDetect(
        rng=rng,
        step=config["metacal"]["step"],
        types=config["metacal"]["types"],
        detect_type=config["sx"]["detect_type"],
        detect_thresh=config["sx"]["detect_thresh"],
        detect_minarea=config["sx"]["detect_minarea"],
        detect_deblend_nthresh=config["sx"]["deblend_nthresh"],
        detect_deblend_cont=config["sx"]["deblend_cont"],
        detect_kernel=config["sx"]["filter_kernel"],
        detect_filter_type=config["sx"]["filter_type"],
        coadd_type=config["coadd"]["type"],
        coadd_multiband=True,
        coadd_fscale=config["coadd"]["fscale"],
        models=[fitter["model"] for fitter in fitters],
        fwhms=[fitter.get("weight", {}).get("fwhm") for fitter in fitters],
        symmetrizes=[fitter.get("symmetrize", True) for fitter in fitters],
        stamp_size=config["meds"]["min_box_size"],
        do_uberseg=(config["meds"].get("weight_type") == "uberseg"),
        mcal_config=config["metacal"],
    )


def _run_sim(
    *,
    seed,
    nband=3,
    models=("wmom",),
    weight_type=None,
):
    """Run metacoadd using the upstream metadetect simulation."""
    rng = np.random.RandomState(seed=seed)
    sim = Sim(
        rng,
        config={
            "nband": nband,
        },
    )
    mbobs = sim.get_mbobs()

    config = _get_metacoadd_config(
        nband=nband,
        models=models,
        weight_type=weight_type,
    )

    return do_metadetect(
        copy.deepcopy(config),
        mbobs,
        rng,
    )


def _check_result_array(result, model, nband):
    """Perform the equivalent upstream result validation."""
    assert set(result) == set(SHEAR_TYPES)

    shape_fields = [
        f"{model}_nimage",
        f"{model}_T",
        f"{model}_Tpsf",
        f"{model}_s2n",
        f"{model}_g1",
        f"{model}_g2",
    ]

    flux_fields = []

    for band in range(nband):
        flux_fields.extend(
            [
                f"{model}_flux_{band}",
                f"{model}_flux_err_{band}",
            ]
        )

    required_fields = [
        f"{model}_flags",
        *shape_fields,
        *flux_fields,
    ]

    for shear in SHEAR_TYPES:
        catalog = result[shear]

        assert catalog is not None
        assert catalog.size > 0

        for field in required_fields:
            assert field in catalog.dtype.names

        good = catalog[f"{model}_flags"] == 0

        if model != "am":
            assert np.any(good), (model, shear)

        if not np.any(good):
            continue

        for field in shape_fields:
            assert np.all(np.isfinite(catalog[field][good])), (
                model,
                shear,
                field,
            )

        assert np.all(catalog[f"{model}_nimage"][good] > 0)
        assert np.all(catalog[f"{model}_T"][good] > 0)
        assert np.all(catalog[f"{model}_Tpsf"][good] > 0)
        assert np.all(np.abs(catalog[f"{model}_g1"][good]) < 1)
        assert np.all(np.abs(catalog[f"{model}_g2"][good]) < 1)

        if model == "am":
            for field in flux_fields:
                assert np.all(np.isnan(catalog[field][good])), (
                    model,
                    shear,
                    field,
                )
        else:
            for field in flux_fields:
                assert np.all(np.isfinite(catalog[field][good])), (
                    model,
                    shear,
                    field,
                )


def _match_catalogs(first, second, max_distance=2.0):
    """Greedily match two catalogs using detection coordinates."""
    first_row = first["sx_row"]
    first_col = first["sx_col"]
    second_row = second["sx_row"]
    second_col = second["sx_col"]

    distance_squared = (first_row[:, None] - second_row[None, :]) ** 2 + (
        first_col[:, None] - second_col[None, :]
    ) ** 2

    first_indices = []
    second_indices = []

    while np.any(np.isfinite(distance_squared)):
        flat_index = np.argmin(distance_squared)
        first_index, second_index = np.unravel_index(
            flat_index,
            distance_squared.shape,
        )

        distance = np.sqrt(distance_squared[first_index, second_index])

        if distance > max_distance:
            break

        first_indices.append(first_index)
        second_indices.append(second_index)

        distance_squared[first_index, :] = np.inf
        distance_squared[:, second_index] = np.inf

    return (
        np.asarray(first_indices, dtype=int),
        np.asarray(second_indices, dtype=int),
    )


def test_detect():
    """Test detection using the upstream metadetect simulation."""
    rng = np.random.RandomState(seed=45)
    mbobs = Sim(rng).get_mbobs()

    config = _get_metacoadd_config(nband=3)
    md = _make_driver(config, rng)

    image, _, weight, _, _ = md.get_coadd_multiband(
        mbobs,
        fscale=config["coadd"]["fscale"],
    )

    catalog, segmentation = md.get_cat(
        image,
        weight,
        wcs=mbobs[0][0].jacobian.get_galsim_wcs(),
    )

    assert catalog.size > 0
    assert segmentation.shape == image.shape
    assert np.any(segmentation > 0)

    for field in [
        "sx_row",
        "sx_col",
        "flux",
        "snr",
    ]:
        assert np.all(np.isfinite(catalog[field]))


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_smoke(model):
    """Adapt the upstream full metadetection smoke test."""
    result = _run_sim(
        seed=116,
        nband=3,
        models=(model,),
    )

    _check_result_array(
        result,
        model=model,
        nband=3,
    )


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_uberseg(model):
    """Adapt the upstream uberseg test."""
    result = _run_sim(
        seed=116,
        nband=3,
        models=(model,),
        weight_type="uberseg",
    )

    _check_result_array(
        result,
        model=model,
        nband=3,
    )


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_zero_weight_all(model):
    """Require rejection when every input weight is zero."""
    rng = np.random.RandomState(seed=53341)
    mbobs = Sim(rng).get_mbobs()

    for obslist in mbobs:
        for obs in obslist:
            obs.set_weight(np.zeros_like(obs.image))

    config = _get_metacoadd_config(
        nband=len(mbobs),
        models=(model,),
    )

    result = do_metadetect(
        config,
        mbobs,
        rng,
    )

    assert result is None


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_zero_weight_some(model):
    """Require rejection when one complete band has zero weight."""
    rng = np.random.RandomState(seed=53341)
    mbobs = Sim(rng).get_mbobs()

    for obs in mbobs[1]:
        obs.set_weight(np.zeros_like(obs.image))

    config = _get_metacoadd_config(
        nband=len(mbobs),
        models=(model,),
    )

    result = do_metadetect(
        config,
        mbobs,
        rng,
    )

    assert result is None


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_nodet_flags_all(model):
    """Require rejection when all bands are masked for detection."""
    rng = np.random.RandomState(seed=53341)
    mbobs = Sim(rng).get_mbobs()

    nodet_flags = MDET_CONFIG["nodet_flags"]

    for obslist in mbobs:
        for obs in obslist:
            obs.set_bmask(
                np.full(
                    obs.image.shape,
                    nodet_flags,
                    dtype=np.int32,
                )
            )

    config = _get_metacoadd_config(
        nband=len(mbobs),
        models=(model,),
    )
    config["nodet_flags"] = nodet_flags

    result = do_metadetect(
        config,
        mbobs,
        rng,
    )

    assert result is None


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_metadetect_nodet_flags_some(model):
    """Require rejection when one band is masked for detection."""
    rng = np.random.RandomState(seed=53341)
    mbobs = Sim(rng).get_mbobs()

    nodet_flags = MDET_CONFIG["nodet_flags"]

    for obs in mbobs[1]:
        obs.set_bmask(
            np.full(
                obs.image.shape,
                nodet_flags,
                dtype=np.int32,
            )
        )

    config = _get_metacoadd_config(
        nband=len(mbobs),
        models=(model,),
    )
    config["nodet_flags"] = nodet_flags

    result = do_metadetect(
        config,
        mbobs,
        rng,
    )

    assert result is None


@pytest.mark.parametrize(
    "mask_region",
    [1, 7],
)
def test_fill_in_mask_col(mask_region):
    """Test mask preservation using upstream mask aggregation."""
    nband = 1
    rng = np.random.RandomState(seed=10)

    mbobs = Sim(
        rng,
        config={"nband": nband},
    ).get_mbobs()

    config = _get_metacoadd_config(
        nband=nband,
        models=("wmom",),
    )
    md = _make_driver(config, rng)

    image, _, weight = md.get_coadd_multiband(
        mbobs,
        fscale=config["coadd"]["fscale"],
    )
    catalog, _ = md.get_cat(
        image,
        weight,
        wcs=mbobs[0][0].jacobian.get_galsim_wcs(),
    )

    assert catalog.size > 0

    # Choose a well-detected object rather than depending on
    # the detector's catalog ordering.
    detection = catalog[np.argmax(catalog["flux"])]

    mask_rng = np.random.RandomState(seed=11)
    full_mask = mask_rng.randint(
        low=0,
        high=64,
        size=mbobs[0][0].image.shape,
        dtype=np.int32,
    )
    mbobs[0][0].set_bmask(full_mask)

    stamp_mbobs = get_stamp_mbobs(
        mbobs,
        detection,
        min_stamp_size=33,
        max_stamp_size=33,
    )

    expected_cutout, cutout_row, cutout_col = get_cutout(
        full_mask,
        detection["x"],
        detection["y"],
        33,
    )

    np.testing.assert_array_equal(
        stamp_mbobs[0][0].bmask,
        expected_cutout,
    )

    expected_value = ngmix_metadetect._fill_in_mask_col(
        mask_region=mask_region,
        rows=np.asarray([detection["y"]]),
        cols=np.asarray([detection["x"]]),
        mask=full_mask,
    )
    measured_value = ngmix_metadetect._fill_in_mask_col(
        mask_region=mask_region,
        rows=np.asarray([cutout_row]),
        cols=np.asarray([cutout_col]),
        mask=stamp_mbobs[0][0].bmask,
    )

    np.testing.assert_array_equal(
        measured_value,
        expected_value,
    )


def test_metadetect_fitter_multi_meas():
    """Adapt the upstream multiple-measurement test."""
    result = _run_sim(
        seed=116,
        nband=3,
        models=tuple(SUPPORTED_MODELS),
    )

    for model in SUPPORTED_MODELS:
        _check_result_array(
            result,
            model=model,
            nband=3,
        )

    for shear in SHEAR_TYPES:
        catalog = result[shear]

        wmom_good = catalog["wmom_flags"] == 0
        pgauss_good = catalog["pgauss_flags"] == 0
        common = wmom_good & pgauss_good

        assert np.any(common)

        assert not np.allclose(
            catalog["wmom_T"][common],
            catalog["pgauss_T"][common],
        )

        wmom_g = np.column_stack(
            [
                catalog["wmom_g1"][common],
                catalog["wmom_g2"][common],
            ]
        )
        pgauss_g = np.column_stack(
            [
                catalog["pgauss_g1"][common],
                catalog["pgauss_g2"][common],
            ]
        )

        assert not np.allclose(wmom_g, pgauss_g)


@pytest.mark.parametrize(
    "model, nband",
    [
        ("wmom", 1),
        ("wmom", 3),
        ("wmom", 4),
        ("pgauss", 1),
        ("pgauss", 3),
        ("pgauss", 4),
    ],
)
def test_metadetect_flux(model, nband):
    """Test fluxes using upstream make_mbobs_sim."""
    flux_factors = np.linspace(1.0, 2.0, nband)

    mbobs = make_mbobs_sim(
        seed=431,
        nband=nband,
        noise_scale=0.2,
        band_flux_factors=flux_factors,
    )

    config = _get_metacoadd_config(
        nband=nband,
        models=(model,),
    )

    result = do_metadetect(
        config,
        mbobs,
        np.random.RandomState(seed=432),
    )

    _check_result_array(
        result,
        model=model,
        nband=nband,
    )

    catalog = result["noshear"]
    good = np.flatnonzero(catalog[f"{model}_flags"] == 0)

    assert good.size > 0

    # The simulation contains one bright source. Select the brightest
    # successful detection in case a noise peak was also detected.
    object_index = good[np.argmax(catalog["flux"][good])]

    fluxes = np.asarray(
        [catalog[f"{model}_flux_{band}"][object_index] for band in range(nband)]
    )

    assert np.all(np.isfinite(fluxes))
    assert np.all(fluxes > 0)

    if nband > 1:
        measured_ratios = fluxes / fluxes[0]
        expected_ratios = flux_factors / flux_factors[0]

        np.testing.assert_allclose(
            measured_ratios,
            expected_ratios,
            rtol=0.20,
            atol=0.05,
        )


def test_metacoadd_is_comparable_to_ngmix_metadetect():
    """Run both drivers on identical upstream simulations."""
    seed = 816

    upstream_rng = np.random.RandomState(seed=seed)
    upstream_mbobs = Sim(upstream_rng).get_mbobs()

    upstream_config = copy.deepcopy(MDET_CONFIG)
    upstream_result = ngmix_metadetect.do_metadetect(
        upstream_config,
        upstream_mbobs,
        upstream_rng,
    )

    metacoadd_rng = np.random.RandomState(seed=seed)
    metacoadd_mbobs = Sim(metacoadd_rng).get_mbobs()

    metacoadd_config = _get_metacoadd_config(
        nband=3,
        models=("wmom",),
    )
    metacoadd_result = do_metadetect(
        metacoadd_config,
        metacoadd_mbobs,
        metacoadd_rng,
    )

    for shear in SHEAR_TYPES:
        reference = upstream_result[shear]
        measured = metacoadd_result[shear]

        assert reference.size > 0
        assert measured.size > 0

        # The detection architectures are not identical, so exact
        # detection counts are not required. They should nevertheless
        # remain of the same order.
        count_ratio = measured.size / reference.size
        assert 0.5 <= count_ratio <= 2.0

        measured_indices, reference_indices = _match_catalogs(
            measured,
            reference,
        )

        minimum_count = min(measured.size, reference.size)
        assert measured_indices.size >= max(
            1,
            minimum_count // 2,
        )

        measured_good = measured["wmom_flags"][measured_indices] == 0
        reference_good = reference["wmom_flags"][reference_indices] == 0
        common = measured_good & reference_good

        assert np.any(common)

        measured_indices = measured_indices[common]
        reference_indices = reference_indices[common]

        np.testing.assert_allclose(
            measured["wmom_T"][measured_indices],
            reference["wmom_T"][reference_indices],
            rtol=0.15,
            atol=0.02,
        )

        measured_g = np.column_stack(
            [
                measured["wmom_g1"][measured_indices],
                measured["wmom_g2"][measured_indices],
            ]
        )
        reference_g = reference["wmom_g"][reference_indices]

        np.testing.assert_allclose(
            measured_g,
            reference_g,
            rtol=0.25,
            atol=0.08,
        )

        np.testing.assert_allclose(
            np.median(measured["wmom_Tpsf"][measured_indices]),
            np.median(reference["psfrec_T"][reference_indices]),
            rtol=0.10,
            atol=0.01,
        )


def test_get_psf_stats():
    """Compare metacoadd PSF size with upstream PSF statistics."""
    rng = np.random.RandomState(seed=10)
    mbobs = Sim(rng).get_mbobs()

    # Upstream stores individual PSF results before combining
    # them into its PSF-statistics dictionary.
    ngmix_fitting.fit_all_psfs(
        mbobs,
        rng,
    )
    upstream_stats = ngmix_metadetect._get_psf_stats(
        mbobs,
        0,
    )

    assert upstream_stats["flags"] == 0
    assert np.isfinite(upstream_stats["g1"])
    assert np.isfinite(upstream_stats["g2"])
    assert np.isfinite(upstream_stats["T"])
    assert upstream_stats["T"] > 0

    config = _get_metacoadd_config(
        nband=len(mbobs),
        models=("wmom",),
    )
    md = _make_driver(
        config,
        np.random.RandomState(seed=11),
    )

    metacoadd_T = md.get_T_psf(mbobs)

    assert np.isfinite(metacoadd_T)
    assert metacoadd_T > 0

    # Upstream uses adaptive PSF moments, whereas metacoadd
    # currently uses its Gaussian PSF runner. The simulated PSF
    # is Gaussian, so the results should still be close.
    np.testing.assert_allclose(
        metacoadd_T,
        upstream_stats["T"],
        rtol=0.05,
        atol=0.01,
    )
