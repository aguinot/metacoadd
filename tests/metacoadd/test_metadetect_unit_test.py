"""
Tests for the metacoadd metadetection driver.

The smoke test follows the approach used by the metadetect test suite:
run all five metacalibration branches on a deterministic simulation and
validate the resulting catalogs.
"""

from unittest.mock import Mock

import numpy as np
import pytest

import ngmix

from galsim_sim import _get_obs
from metacoadd import metadetect as mdet
from metacoadd.detect import DET_CAT_DTYPE
from metacoadd.imcom_maps import (
    get_imcom_map_cat_dtype,
)
from metacoadd.metadetect import (
    MetaDetect,
    do_metadetect,
    get_shape_cat_dtype,
)
from metacoadd.ori_img_process import (
    get_original_cat_dtype,
)


MCAL_TYPES = [
    "noshear",
    "1p",
    "1m",
    "2p",
    "2m",
]

MCAL_CONFIG = {
    "step": 0.01,
    "types": MCAL_TYPES,
    "psf": "fitgauss",
    "use_noise_image": True,
    "fixnoise": True,
    "has_pixel": False,
}


def _make_mbobs(nband=2, seed=10):
    """Build deterministic multiband observations."""
    mbobs = ngmix.MultiBandObsList()

    for band_index in range(nband):
        obs = _get_obs(
            np.random.RandomState(seed + band_index),
            noise=1.0e-4,
            set_noise_image=True,
            n=35,
        )
        obs.set_ormask(
            np.zeros_like(
                obs.image,
                dtype=np.int32,
            )
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _make_md(**kwargs):
    """Build a MetaDetect instance with a complete configuration."""
    config = {
        "rng": np.random.RandomState(20),
        "models": ["wmom"],
        "fwhms": [1.2],
        "symmetrizes": [True],
        "stamp_size": 35,
        "coadd_fscale": [[1.0], [1.0]],
        "mcal_config": MCAL_CONFIG.copy(),
    }
    config.update(kwargs)
    return MetaDetect(**config)


def _make_sep_cat(n_obj=1):
    """Build a minimal detection catalog."""
    cat = np.zeros(n_obj, dtype=DET_CAT_DTYPE)

    if n_obj > 0:
        cat["number"] = np.arange(1, n_obj + 1)
        cat["x"] = 17.0
        cat["y"] = 17.0
        cat["xx"] = 2.0
        cat["yy"] = 2.0
        cat["xy"] = 0.0

    return cat


def _shape_result(nband, offset=0.0):
    """Build a complete synthetic shape result."""
    return {
        "flags": 0,
        "nimage": 1,
        "pars": np.asarray(
            [
                0.1 + offset,
                0.2 + offset,
                0.01,
                -0.02,
                1.5,
                10.0,
            ]
        ),
        "T": 1.5 + offset,
        "T_err": 0.1,
        "Tr": 1.2,
        "Tpsf": 0.7,
        "rho4": 2.0,
        "rho4_err": 0.2,
        "s2n": 50.0,
        "e1": 0.01,
        "e2": -0.02,
        "e1err": 0.001,
        "e2err": 0.002,
        "g1": 0.005,
        "g2": -0.01,
        "flux": np.arange(1, nband + 1) * 10.0 + offset,
        "flux_err": np.arange(1, nband + 1) + offset,
    }


def test_get_shape_cat_dtype():
    """Test shape catalog field names and dtypes."""
    dtype = np.dtype(get_shape_cat_dtype("wmom"))

    expected_names = (
        "wmom_flags",
        "wmom_nimage",
        "wmom_dx",
        "wmom_dy",
        "wmom_T",
        "wmom_T_err",
        "wmom_Tr",
        "wmom_Tpsf",
        "wmom_rho4",
        "wmom_rho4_err",
        "wmom_s2n",
        "wmom_e1",
        "wmom_e2",
        "wmom_e1err",
        "wmom_e2err",
        "wmom_g1",
        "wmom_g2",
    )

    assert dtype.names == expected_names
    assert dtype["wmom_flags"] == np.dtype(np.int32)
    assert dtype["wmom_nimage"] == np.dtype(np.int32)

    for name in expected_names[2:]:
        assert dtype[name] == np.dtype(np.float64)


def test_init_defaults_and_stamp_size():
    """Test default metacal settings and odd stamp-size enforcement."""
    md = MetaDetect(
        rng=np.random.RandomState(30),
        models=["wmom"],
        fwhms=[1.2],
        symmetrizes=[True],
        stamp_size=34,
        coadd_fscale=[[1.0]],
    )

    assert md._stamp_size == 35
    assert md.mcal_config["step"] == 0.01
    assert md.mcal_config["types"] == MCAL_TYPES
    assert md.mcal_config["fixnoise"] is True
    assert md.mcal_config["use_noise_image"] is True
    assert md.mcal_config["has_pixel"] is False

    with pytest.raises(
        ValueError,
        match="mcal_config must be a dictionary",
    ):
        MetaDetect(
            rng=np.random.RandomState(31),
            mcal_config="invalid",
        )


def test_init_optional_catalog_validation():
    """Test original-image and IMCOM configuration validation."""
    base = {
        "rng": np.random.RandomState(40),
        "mcal_config": MCAL_CONFIG.copy(),
    }

    with pytest.raises(
        ValueError,
        match="original_image_config must contain",
    ):
        MetaDetect(
            **base,
            original_image_config={},
        )

    with pytest.raises(
        ValueError,
        match="original_image_config 'mcal_type_ref'",
    ):
        MetaDetect(
            **base,
            original_image_config={
                "mcal_type_ref": "invalid",
            },
        )

    with pytest.raises(
        ValueError,
        match="imcom_map_config must contain",
    ):
        MetaDetect(
            **base,
            imcom_map_config={
                "layers": {},
            },
        )

    with pytest.raises(
        ValueError,
        match="imcom_map_config 'mcal_type_ref'",
    ):
        MetaDetect(
            **base,
            imcom_map_config={
                "mcal_type_ref": "invalid",
                "layers": {},
            },
        )

    md = MetaDetect(
        **base,
        original_image_config={
            "mcal_type_ref": "noshear",
        },
        imcom_map_config={
            "mcal_type_ref": "noshear",
            "layers": {
                "Coverage": {},
            },
        },
    )

    assert md._do_original_image is True
    assert md._original_image_mcal_type_ref == "noshear"
    assert md._do_imcom_map is True
    assert md._imcom_map_mcal_type_ref == "noshear"


def test_get_metacal(monkeypatch):
    """Test delegation to the metacalibration handler."""
    mbobs = _make_mbobs()
    expected = {"noshear": mbobs}

    handler = Mock()
    handler.get_all.return_value = expected
    handler_class = Mock(return_value=handler)

    monkeypatch.setattr(
        mdet,
        "MetacalHandler",
        handler_class,
    )

    md = _make_md()
    result = md.get_metacal(mbobs)

    assert result is expected

    handler_class.assert_called_once_with(
        rng=md.rng,
        fixnoise=True,
        use_noise_image=True,
        mcal_config={
            "step": 0.01,
            "has_pixel": False,
        },
    )
    handler.get_all.assert_called_once_with(
        mbobs,
        MCAL_TYPES,
    )


def test_set_power_spectrum(monkeypatch):
    """Test Fourier power-spectrum creation and caching."""
    mbobs = _make_mbobs()
    existing_ps = np.asarray([42.0])
    mbobs[1][0].ps = existing_ps

    estimate_ps = Mock(
        return_value=np.full((5, 3), 4.0),
    )
    monkeypatch.setattr(
        mdet,
        "estimate_noise_ps_analytic",
        estimate_ps,
    )

    md = _make_md()
    md.gal_runners = {"wmom": Mock()}
    md._set_power_spectrum(mbobs)

    estimate_ps.assert_not_called()
    assert not hasattr(mbobs[0][0], "ps")

    md.gal_runners = {
        "wmom": Mock(),
        "fourier_gauss": Mock(),
    }
    md._set_power_spectrum(mbobs)

    estimate_ps.assert_called_once()
    noise, stamp_size = estimate_ps.call_args.args

    np.testing.assert_array_equal(
        noise,
        mbobs[0][0].noise,
    )
    assert stamp_size == 35

    np.testing.assert_array_equal(
        mbobs[0][0].ps,
        np.full((5, 3), 4.0),
    )
    assert mbobs[1][0].ps is existing_ps


def test_get_T_psf(monkeypatch):
    """Test weighted averaging of PSF sizes."""
    mbobs = _make_mbobs()

    weight_band_0 = np.full_like(
        mbobs[0][0].image,
        2.0,
        dtype=np.float64,
    )
    weight_band_0[0, 0] = 0.0
    weight_band_0[0, 1] = 4.0

    weight_band_1 = np.full_like(
        mbobs[1][0].image,
        6.0,
        dtype=np.float64,
    )
    weight_band_1[0, 0] = 0.0

    mbobs[0][0].set_weight(weight_band_0)
    mbobs[1][0].set_weight(weight_band_1)

    psf_runner = Mock()
    psf_runner.go.side_effect = [
        {"T": 1.0},
        {"T": 3.0},
    ]

    get_psf_runner = Mock(return_value=psf_runner)
    monkeypatch.setattr(
        mdet,
        "get_gauss_psf_runner",
        get_psf_runner,
    )

    md = _make_md()
    result = md.get_T_psf(mbobs)

    # Median positive weights are 2 and 6.
    expected = (1.0 * 2.0 + 3.0 * 6.0) / 8.0

    np.testing.assert_allclose(result, expected)

    get_psf_runner.assert_called_once_with(md.rng)
    assert psf_runner.go.call_count == 2
    assert psf_runner.go.call_args_list[0].args[0] is mbobs[0][0].psf
    assert psf_runner.go.call_args_list[1].args[0] is mbobs[1][0].psf


def test_coadd_and_detection_delegation(monkeypatch):
    """Test coaddition and detection configuration forwarding."""
    mbobs = _make_mbobs()
    expected_coadd = (
        np.ones((3, 3)),
        np.full((3, 3), 2.0),
        np.full((3, 3), 3.0),
    )

    coadd = Mock()
    coadd.make.return_value = expected_coadd
    coadd_class = Mock(return_value=coadd)
    get_coadd = Mock(return_value=coadd_class)

    monkeypatch.setattr(
        mdet,
        "get_coadd_class",
        get_coadd,
    )

    md = _make_md(
        coadd_type="weighted-average",
        detect_type="absolute",
        detect_thresh=7.0,
        detect_minarea=9,
        detect_deblend_nthresh=16,
        detect_deblend_cont=0.02,
        detect_kernel=[[1.0]],
        detect_filter_type="match",
    )

    result = md.get_coadd_multiband(
        mbobs,
        fscale=[[1.0], [2.0]],
        target_zp=31.0,
    )

    assert len(result) == len(expected_coadd)
    for result_array, expected_array in zip(
        result,
        expected_coadd,
        strict=True,
    ):
        np.testing.assert_array_equal(
            result_array,
            expected_array,
        )

    get_coadd.assert_called_once_with("weighted-average")
    coadd_class.assert_called_once_with(
        mbobs,
        fscale=[[1.0], [2.0]],
        zeropoints=None,
        target_zp=31.0,
    )
    coadd.make.assert_called_once_with()

    expected_cat = _make_sep_cat()
    expected_seg = np.ones((3, 3), dtype=np.int32)
    detect = Mock(
        return_value=(expected_cat, expected_seg),
    )
    monkeypatch.setattr(mdet, "get_cat", detect)

    image = np.ones((3, 3))
    weight = np.full((3, 3), 4.0)
    wcs = object()

    cat, seg = md.get_cat(
        image,
        weight,
        wcs=wcs,
    )

    assert cat is expected_cat
    assert seg is expected_seg

    detect.assert_called_once_with(
        image,
        weight,
        thresh_type="absolute",
        thresh=7.0,
        minarea=9,
        deblend_nthresh=16,
        deblend_cont=0.02,
        kernel=[[1.0]],
        filter_type="match",
        wcs=wcs,
    )


def test_get_shape_cat(monkeypatch):
    """Test cutout extraction and g/e result normalization."""
    mbobs = _make_mbobs()
    sep_cat = _make_sep_cat(n_obj=2)
    seg_map = np.ones((35, 35), dtype=np.int32)
    stamp_mbobs = object()

    get_stamp = Mock(return_value=stamp_mbobs)
    monkeypatch.setattr(
        mdet,
        "get_stamp_mbobs",
        get_stamp,
    )

    g_runner = Mock()
    g_runner.go.return_value = {
        "flags": 0,
        "g": np.asarray([0.1, -0.2]),
        "T": 1.5,
    }

    e_runner = Mock()
    e_runner.go.return_value = {
        "flags": 0,
        "e": np.asarray([0.3, 0.4]),
        "T": 2.0,
    }

    md = _make_md()
    md.gal_runners = {
        "gauss": g_runner,
        "wmom": e_runner,
    }

    result = md.get_shape_cat(
        mbobs,
        sep_cat,
        seg_map,
        T_psf=0.7,
        do_uberseg=True,
    )

    assert set(result) == {"gauss", "wmom"}
    assert len(result["gauss"]) == 2
    assert len(result["wmom"]) == 2

    for res in result["gauss"]:
        assert res["g1"] == 0.1
        assert res["g2"] == -0.2
        assert res["Tpsf"] == 0.7

    for res in result["wmom"]:
        assert res["g1"] == 0.3
        assert res["g2"] == 0.4
        assert res["Tpsf"] == 0.7

    assert get_stamp.call_count == 2

    for call in get_stamp.call_args_list:
        assert call.args[0] is mbobs
        assert call.kwargs["min_stamp_size"] == 35
        assert call.kwargs["max_stamp_size"] == 35
        assert call.kwargs["do_uberseg"] is True
        assert call.kwargs["seg_map"] is seg_map

    assert g_runner.go.call_count == 2
    assert e_runner.go.call_count == 2


def test_build_output_cat():
    """Test merging detection, shape, original, and IMCOM results."""
    md = _make_md(
        models=["wmom", "fourier_gauss"],
        fwhms=[1.2, None],
        symmetrizes=[True, True],
    )
    md._nband = 2
    md.gal_runners = {
        "wmom": Mock(),
        "fourier_gauss": Mock(),
    }
    md._imcom_map_config = {
        "layers": {
            "Coverage": {},
        }
    }

    sep_cat = _make_sep_cat()
    sep_cat["number"] = 7
    sep_cat["x"] = 12.5
    sep_cat["y"] = 13.5

    original_cat = np.zeros(
        1,
        dtype=get_original_cat_dtype(2),
    )
    original_cat["original_flux_1"] = 91.0

    imcom_cat = np.zeros(
        1,
        dtype=get_imcom_map_cat_dtype(
            md._imcom_map_config["layers"],
            2,
        ),
    )
    imcom_cat["IMCOM_Coverage_mean_0"] = 0.8
    imcom_cat["IMCOM_Coverage_std_1"] = 0.04

    wmom_result = _shape_result(2)
    fourier_result = _shape_result(2, offset=1.0)

    # Exercise the broad missing-result fallback.
    del fourier_result["rho4_err"]

    shape_cat = {
        "wmom": [wmom_result],
        "fourier_gauss": [fourier_result],
    }

    result = md.build_output_cat(
        sep_cat,
        shape_cat,
        original_cat=original_cat,
        imcom_map_cat=imcom_cat,
    )

    assert result.shape == (1,)
    assert result["number"][0] == 7
    assert result["x"][0] == 12.5
    assert result["y"][0] == 13.5

    assert result["wmom_flags"][0] == 0
    assert result["wmom_dx"][0] == 0.1
    assert result["wmom_dy"][0] == 0.2
    assert result["wmom_T"][0] == 1.5
    assert result["wmom_flux_0"][0] == 10.0
    assert result["wmom_flux_1"][0] == 20.0
    assert result["wmom_flux_err_1"][0] == 2.0

    assert result["fourier_gauss_dx"][0] == 1.1
    assert result["fourier_gauss_flux_0"][0] == 11.0

    # Missing rho4_err remains at its zero initialization.
    assert result["fourier_gauss_rho4_err"][0] == 0.0

    assert result["original_flux_1"][0] == 91.0
    assert result["IMCOM_Coverage_mean_0"][0] == 0.8
    assert result["IMCOM_Coverage_std_1"][0] == 0.04


def test_go_input_and_coadd_validation(monkeypatch):
    """Test input type and coadd configuration validation."""
    get_fitters = Mock(return_value={})
    monkeypatch.setattr(
        mdet,
        "get_fitters",
        get_fitters,
    )

    md = _make_md()

    with pytest.raises(
        ValueError,
        match="mb_obs must be an instance",
    ):
        md.go(object())

    mbobs = _make_mbobs()

    invalid_configs = [
        {
            "coadd_fscale": None,
            "coadd_zeropoints": None,
            "match": "Either coadd_zeropoints or coadd_fscale",
        },
        {
            "coadd_fscale": [[1.0], [1.0]],
            "coadd_zeropoints": [[30.0], [30.0]],
            "match": "Either coadd_zeropoints or coadd_fscale",
        },
        {
            "coadd_fscale": [[1.0]],
            "coadd_zeropoints": None,
            "match": "coadd_fscale must have the same length",
        },
        {
            "coadd_fscale": None,
            "coadd_zeropoints": [[30.0]],
            "match": "coadd_zeropoints must have the same length",
        },
    ]

    for config in invalid_configs:
        md = _make_md(
            coadd_fscale=config["coadd_fscale"],
            coadd_zeropoints=config["coadd_zeropoints"],
        )
        md.get_metacal = Mock(return_value={})

        with pytest.raises(ValueError, match=config["match"]):
            md.go(mbobs)


def test_go_without_multiband_coadd(monkeypatch):
    """Test detection directly from the first input observation."""
    mcal_config = MCAL_CONFIG.copy()
    mcal_config["types"] = ["noshear"]

    mbobs = _make_mbobs()
    sep_cat = _make_sep_cat()
    seg_map = np.ones_like(
        mbobs[0][0].image,
        dtype=np.int32,
    )
    shape_cat = {
        "wmom": [_shape_result(nband=2)],
    }
    expected = np.zeros(
        1,
        dtype=[("result", np.int32)],
    )

    get_fitters = Mock(
        return_value={
            "wmom": Mock(),
        }
    )
    monkeypatch.setattr(
        mdet,
        "get_fitters",
        get_fitters,
    )

    md = _make_md(
        coadd_multiband=False,
        mcal_config=mcal_config,
    )
    md.get_metacal = Mock(
        return_value={
            "noshear": mbobs,
        }
    )
    md.get_T_psf = Mock(return_value=0.7)
    md.get_coadd_multiband = Mock()
    md.get_cat = Mock(
        return_value=(sep_cat, seg_map),
    )
    md.get_shape_cat = Mock(
        return_value=shape_cat,
    )
    md.build_output_cat = Mock(
        return_value=expected,
    )
    md._set_power_spectrum = Mock()

    result = md.go(mbobs)

    assert result["noshear"] is expected

    # No detection coadd should be constructed.
    md.get_coadd_multiband.assert_not_called()

    # Detection uses the image and weight from the first band and
    # first observation directly.
    md.get_cat.assert_called_once()
    detect_call = md.get_cat.call_args

    np.testing.assert_array_equal(
        detect_call.args[0],
        mbobs[0][0].image,
    )
    np.testing.assert_array_equal(
        detect_call.args[1],
        mbobs[0][0].weight,
    )
    assert detect_call.kwargs["wcs"] == mbobs[0][0].jacobian.get_galsim_wcs()

    md.get_shape_cat.assert_called_once_with(
        mbobs,
        sep_cat,
        seg_map,
        0.7,
        do_uberseg=False,
    )
    md._set_power_spectrum.assert_not_called()

    md.build_output_cat.assert_called_once_with(
        sep_cat,
        shape_cat,
        None,
        None,
    )


def test_go_original_and_imcom_catalogs(monkeypatch):
    """Test reference and placeholder optional catalog branches."""
    mcal_types = ["noshear", "1p"]
    mcal_config = MCAL_CONFIG.copy()
    mcal_config["types"] = mcal_types

    original_config = {
        "mcal_type_ref": "noshear",
    }
    imcom_config = {
        "mcal_type_ref": "noshear",
        "layers": {
            "Coverage": {},
        },
    }

    mbobs = _make_mbobs()
    sep_cat = _make_sep_cat()
    seg_map = np.ones_like(
        mbobs[0][0].image,
        dtype=np.int32,
    )
    shape_cat = {
        "wmom": [_shape_result(nband=2)],
    }

    original_reference = np.zeros(
        1,
        dtype=get_original_cat_dtype(2),
    )
    original_reference["original_flux_0"] = 15.0
    original_reference["original_flux_1"] = 25.0

    imcom_reference = np.zeros(
        1,
        dtype=get_imcom_map_cat_dtype(
            imcom_config["layers"],
            2,
        ),
    )
    imcom_reference["IMCOM_Coverage_mean_0"] = 0.8
    imcom_reference["IMCOM_Coverage_mean_1"] = 0.9

    def _make_original_placeholder(n_obj, nband):
        return np.zeros(
            n_obj,
            dtype=get_original_cat_dtype(nband),
        )

    def _make_imcom_placeholder(
        n_obj,
        imcom_map_config,
        nband,
    ):
        return np.zeros(
            n_obj,
            dtype=get_imcom_map_cat_dtype(
                imcom_map_config,
                nband,
            ),
        )

    get_original = Mock(
        return_value=original_reference,
    )
    get_original_output = Mock(
        side_effect=_make_original_placeholder,
    )
    extract_imcom = Mock(
        return_value=imcom_reference,
    )
    get_imcom_output = Mock(
        side_effect=_make_imcom_placeholder,
    )

    monkeypatch.setattr(
        mdet,
        "get_original_image_cat",
        get_original,
    )
    monkeypatch.setattr(
        mdet,
        "get_output_original_cat",
        get_original_output,
    )
    monkeypatch.setattr(
        mdet,
        "extract_imcom_maps",
        extract_imcom,
    )
    monkeypatch.setattr(
        mdet,
        "get_output_imcom_map_cat",
        get_imcom_output,
    )
    monkeypatch.setattr(
        mdet,
        "get_fitters",
        Mock(
            return_value={
                "wmom": Mock(),
            }
        ),
    )

    noshear_output = np.zeros(
        1,
        dtype=[("result", np.int32)],
    )
    one_plus_output = np.ones(
        1,
        dtype=[("result", np.int32)],
    )

    md = _make_md(
        mcal_config=mcal_config,
        original_image_config=original_config,
        imcom_map_config=imcom_config,
        do_uberseg=True,
    )
    md.get_metacal = Mock(
        return_value={
            "noshear": mbobs,
            "1p": mbobs,
        }
    )
    md.get_T_psf = Mock(return_value=0.7)
    md.get_coadd_multiband = Mock(
        return_value=(
            np.ones_like(mbobs[0][0].image),
            np.ones_like(mbobs[0][0].image),
            np.ones_like(mbobs[0][0].weight),
        )
    )
    md.get_cat = Mock(
        return_value=(sep_cat, seg_map),
    )
    md.get_shape_cat = Mock(
        return_value=shape_cat,
    )
    md.build_output_cat = Mock(
        side_effect=[
            noshear_output,
            one_plus_output,
        ]
    )

    result = md.go(mbobs)

    assert result["noshear"] is noshear_output
    assert result["1p"] is one_plus_output

    # The original images are measured only for the reference
    # metacalibration branch.
    get_original.assert_called_once_with(
        md.rng,
        mbobs,
        sep_cat,
        seg_map=seg_map,
        cutout_size=35,
        do_uberseg=True,
        wmom_fwhm=0.5,
        psf_fitter="gauss",
    )

    # The non-reference branch receives an output placeholder.
    get_original_output.assert_called_once_with(
        1,
        2,
    )

    # The IMCOM maps are extracted only for the reference branch.
    extract_imcom.assert_called_once_with(
        mbobs,
        sep_cat,
        seg_map,
        imcom_config,
    )

    # The non-reference branch receives an IMCOM placeholder.
    get_imcom_output.assert_called_once_with(
        1,
        imcom_config["layers"],
        2,
    )

    assert md.build_output_cat.call_count == 2
    noshear_call, one_plus_call = md.build_output_cat.call_args_list

    # noshear is the configured reference branch.
    assert noshear_call.args[0] is sep_cat
    assert noshear_call.args[1] is shape_cat
    assert noshear_call.args[2] is original_reference
    assert noshear_call.args[3] is imcom_reference

    # 1p contains the generated placeholders.
    original_placeholder = one_plus_call.args[2]
    imcom_placeholder = one_plus_call.args[3]

    assert original_placeholder is not original_reference
    assert imcom_placeholder is not imcom_reference

    # The original-image placeholder flags are explicitly set to 42.
    for band_index in range(2):
        np.testing.assert_array_equal(
            original_placeholder[f"original_flags_{band_index}"],
            [42],
        )

    # The mocked IMCOM placeholder remains zero initialized.
    for name in imcom_placeholder.dtype.names:
        np.testing.assert_array_equal(
            imcom_placeholder[name],
            [0.0],
        )

    assert md.get_coadd_multiband.call_count == 2
    assert md.get_cat.call_count == 2
    assert md.get_shape_cat.call_count == 2


def test_do_metadetect_configuration(monkeypatch):
    """Test translation of the public configuration dictionary."""
    mbobs = _make_mbobs(nband=1)
    rng = np.random.RandomState(100)
    expected = {"noshear": np.zeros(0)}

    md = Mock()
    md.go.return_value = expected
    md_class = Mock(return_value=md)

    monkeypatch.setattr(
        mdet,
        "MetaDetect",
        md_class,
    )

    config = {
        "fitters": [
            {
                "model": "wmom",
                "weight": {"fwhm": 1.2},
                "symmetrize": False,
            },
            {
                "model": "gauss",
            },
        ],
        "metacal": {
            "step": 0.02,
            "types": ["noshear"],
            "psf": "fitgauss",
            "use_noise_image": True,
            "fixnoise": False,
            "has_pixel": True,
        },
        "sx": {
            "detect_type": "absolute",
            "detect_thresh": 5.0,
            "detect_minarea": 7,
            "deblend_nthresh": 8,
            "deblend_cont": 0.01,
            "filter_kernel": [[1.0]],
            "filter_type": "match",
        },
        "coadd": {
            "type": "median",
            "fscale": [[1.0]],
            "target_zp": 31.0,
        },
        "meds": {
            "min_box_size": 34,
            "weight_type": "uberseg",
        },
    }

    result = do_metadetect(
        config=config,
        mbobs=mbobs,
        rng=rng,
    )

    assert result is expected
    assert md_class.call_args.args == (rng,)

    kwargs = md_class.call_args.kwargs

    assert kwargs["step"] == 0.02
    assert kwargs["types"] == ["noshear"]
    assert kwargs["detect_type"] == "absolute"
    assert kwargs["detect_thresh"] == 5.0
    assert kwargs["detect_minarea"] == 7
    assert kwargs["detect_filter_type"] == "match"
    assert kwargs["coadd_type"] == "median"
    assert kwargs["coadd_fscale"] == [[1.0]]
    assert kwargs["models"] == ["wmom", "gauss"]
    assert kwargs["fwhms"] == [1.2, None]
    assert kwargs["symmetrizes"] == [False, True]
    assert kwargs["stamp_size"] == 34
    assert kwargs["do_uberseg"] is True
    assert kwargs["mcal_config"] == config["metacal"]

    md.go.assert_called_once_with(mbobs)


def test_metadetect_smoke():
    """Run a seeded one-band metadetection smoke test."""
    mbobs = _make_mbobs(
        nband=1,
        seed=116,
    )

    config = {
        "fitters": [
            {
                "model": "wmom",
                "weight": {
                    "fwhm": 1.2,
                },
                "symmetrize": True,
            },
        ],
        "metacal": {
            "step": 0.01,
            "types": MCAL_TYPES,
            "psf": "fitgauss",
            "use_noise_image": True,
            "fixnoise": True,
            "has_pixel": False,
        },
        "sx": {
            "detect_type": "relative",
            "detect_thresh": 1.5,
            "detect_minarea": 5,
            "deblend_nthresh": 32,
            "deblend_cont": 0.005,
            "filter_kernel": [[1.0]],
            "filter_type": "conv",
        },
        "coadd": {
            "type": "average",
            "fscale": [[1.0]],
            "target_zp": 30.0,
        },
        "meds": {
            "min_box_size": 35,
            "weight_type": None,
        },
    }

    result = do_metadetect(
        config=config,
        mbobs=mbobs,
        rng=np.random.RandomState(117),
    )

    assert set(result) == set(MCAL_TYPES)

    for shear_type in MCAL_TYPES:
        shear_result = result[shear_type]

        assert shear_result.size > 0
        assert "wmom_flags" in shear_result.dtype.names
        assert "wmom_T" in shear_result.dtype.names
        assert "wmom_g1" in shear_result.dtype.names
        assert "wmom_g2" in shear_result.dtype.names
        assert "wmom_flux_0" in shear_result.dtype.names
        assert "wmom_flux_err_0" in shear_result.dtype.names

        successful = shear_result["wmom_flags"] == 0
        assert np.any(successful)

        assert np.all(np.isfinite(shear_result["wmom_T"][successful]))
        assert np.all(np.isfinite(shear_result["wmom_flux_0"][successful]))
