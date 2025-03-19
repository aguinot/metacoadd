import copy

import numpy as np

import sep

from astropy.wcs import WCS

from sf_tools.image.stamp import FetchStamps

DET_CAT_DTYPE = [
    ("number", np.int64),
    ("ra", np.float64),
    ("dec", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("a", np.float64),
    ("b", np.float64),
    ("elongation", np.float64),
    ("ellipticity", np.float64),
    ("kronrad", np.float64),
    ("flux", np.float64),
    ("flux_err", np.float64),
    ("flux_radius", np.float64),
    ("snr", np.float64),
    ("flags", np.int64),
    ("ext_flags", np.int64),
]


def get_cutout(img, x, y, stamp_size):
    fs = FetchStamps(img, int(stamp_size / 2))
    x_round = np.round(x).astype(int)
    y_round = np.round(y).astype(int)
    dx = x_round - x
    dy = y_round - y
    fs.get_pixels(np.array([[y_round, x_round]]))
    vign = fs.scan()[0].astype(np.float64)

    return vign, dx, dy


def get_output_cat(n_obj):
    out = np.array(
        list(map(tuple, np.zeros((len(DET_CAT_DTYPE), n_obj)).T)),
        dtype=DET_CAT_DTYPE,
    )
    return out


def get_cat(img, weight, thresh=1.5, header=None, wcs=None, mask=None):
    # NOTE: Might need to look again into this. For now we keep it simple.
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0

    rms = np.median(np.sqrt(1 / weight[m]))

    if (header is not None) and (wcs is not None):
        raise ValueError("Only one of header or wcs can be provided.")
    elif header is not None:
        wcs = WCS(header)

    obj, seg = sep.extract(
        img,
        thresh,
        err=rms,
        segmentation_map=True,
        minarea=5,
        deblend_nthresh=32,
        deblend_cont=0.001,
    )
    n_obj = len(obj)
    seg_id = np.arange(1, n_obj + 1, dtype=np.int32)

    kronrads, krflags = sep.kron_radius(
        img,
        obj["x"],
        obj["y"],
        obj["a"],
        obj["b"],
        obj["theta"],
        6.0,
        seg_id=seg_id,
        segmap=seg,
        mask=mask_rms,
    )
    fluxes = np.ones(n_obj) * -10.0
    fluxerrs = np.ones(n_obj) * -10.0
    flux_rad = np.ones(n_obj) * -10.0
    snr = np.ones(n_obj) * -10.0
    flags = np.ones(n_obj, dtype=np.int64) * 64
    flags_rad = np.ones(n_obj, dtype=np.int64) * 64

    good_flux = (
        (kronrads > 0)
        & (obj["b"] > 0)
        & (obj["a"] >= obj["b"])
        & (obj["theta"] >= -np.pi / 2)
        & (obj["theta"] <= np.pi / 2)
    )
    fluxes[good_flux], fluxerrs[good_flux], flags[good_flux] = sep.sum_ellipse(
        img,
        obj["x"][good_flux],
        obj["y"][good_flux],
        obj["a"][good_flux],
        obj["b"][good_flux],
        obj["theta"][good_flux],
        2.5 * kronrads[good_flux],
        err=rms,
        subpix=1,
        seg_id=seg_id[good_flux],
        segmap=seg,
        mask=mask_rms,
    )

    flux_rad[good_flux], flags_rad[good_flux] = sep.flux_radius(
        img,
        obj["x"][good_flux],
        obj["y"][good_flux],
        6.0 * obj["a"][good_flux],
        0.5,
        normflux=fluxes[good_flux],
        subpix=1,
        seg_id=seg_id[good_flux],
        segmap=seg,
        mask=mask_rms,
    )

    good_snr = (fluxes > 0) & (fluxerrs > 0)
    snr[good_snr] = fluxes[good_snr] / fluxerrs[good_snr]

    ra, dec = wcs.all_pix2world(obj["x"] - 1, obj["y"] - 1, 0)

    # Build the equivalent to IMAFLAGS_ISO
    # But you only know if the object is flagged or not, you don't get the flag
    ext_flags = np.zeros(n_obj, dtype=int)
    if mask is not None:
        for i, seg_id_tmp in enumerate(seg_id):
            seg_map_tmp = copy.deepcopy(seg)
            seg_map_tmp[seg_map_tmp != seg_id_tmp] = 0
            check_map = seg_map_tmp + mask
            if (check_map > seg_id_tmp).any():
                ext_flags[i] = 1

    out = get_output_cat(n_obj)

    out["number"] = seg_id
    out["ra"] = ra
    out["dec"] = dec
    out["x"] = obj["x"] - 1
    out["y"] = obj["y"] - 1
    out["a"] = obj["a"]
    out["b"] = obj["b"]
    out["elongation"] = obj["a"] / obj["b"]
    out["ellipticity"] = 1.0 - obj["b"] / obj["a"]
    out["kronrad"] = kronrads
    out["flux"] = fluxes
    out["flux_err"] = fluxerrs
    out["flux_radius"] = flux_rad
    out["snr"] = snr
    out["flags"] = obj["flag"] | krflags | flags | flags_rad
    out["ext_flags"] = ext_flags

    return out, seg
