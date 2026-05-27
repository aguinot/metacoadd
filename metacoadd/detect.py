import copy

from math import sqrt

import numpy as np
import numba as nb

from ngmix import Observation, ObsList, MultiBandObsList, Jacobian

import sep

from astropy.wcs import WCS

from .uberseg import fast_uberseg

DES_KERNEL = np.array(
    [
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
    ]
)

DET_CAT_DTYPE = [
    ("number", np.int64),
    ("npix", np.int64),
    ("ra", np.float64),
    ("dec", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("sx_row", np.float64),
    ("sx_col", np.float64),
    ("a", np.float64),
    ("b", np.float64),
    ("xx", np.float64),
    ("yy", np.float64),
    ("xy", np.float64),
    ("elongation", np.float64),
    ("ellipticity", np.float64),
    ("kronrad", np.float64),
    ("flux", np.float64),
    ("flux_err", np.float64),
    ("flux_radius", np.float64),
    ("snr", np.float64),
    ("flags", np.int64),
    ("flux_flags", np.int64),
    ("ext_flags", np.int64),
]


@nb.njit(fastmath=True, cache=True)
def get_cutout_size(Qxx, Qxy, Qyy, n_sigma=3.0):
    # Compute trace and determinant
    trace = Qxx + Qyy

    # Compute eigenvalues analytically for symmetric 2x2 matrix
    temp = sqrt((Qxx - Qyy) ** 2 + 4 * Qxy**2)
    lam1 = 0.5 * (trace + temp)
    lam2 = 0.5 * (trace - temp)

    lam_max = max(lam1, lam2)

    return 2.0 * n_sigma * sqrt(lam_max)


def get_cutout(img, x, y, stamp_size):
    orow = int(y)
    ocol = int(x)
    half_box_size = stamp_size // 2
    maxrow, maxcol = img.shape

    ostart_row = orow - half_box_size + 1
    ostart_col = ocol - half_box_size + 1
    oend_row = orow + half_box_size + 2  # plus one for slices
    oend_col = ocol + half_box_size + 2

    ostart_row = max(0, ostart_row)
    ostart_col = max(0, ostart_col)
    oend_row = min(maxrow, oend_row)
    oend_col = min(maxcol, oend_col)

    cutout_row = y - ostart_row
    cutout_col = x - ostart_col

    return (
        img[ostart_row:oend_row, ostart_col:oend_col],
        cutout_row,
        cutout_col,
    )


def get_stamp_mbobs(
    img_mbobs,
    det_row,
    min_stamp_size=71,
    max_stamp_size=201,
    do_uberseg=False,
    seg_map=None,
):

    if do_uberseg and seg_map is None:
        raise ValueError("seg_map must be provided if do_uberseg is True.")

    # Get stamp size
    # stamp_size = np.int64(np.ceil(np.sqrt(det_row["npix"] / np.pi) * 2))
    stamp_size = get_cutout_size(
        det_row["xx"],
        det_row["xy"],
        det_row["yy"],
        n_sigma=5.0,
    )
    stamp_size = np.int64(np.ceil(stamp_size))
    if stamp_size % 2 == 0:
        stamp_size += 1
    stamp_size = max(min_stamp_size, stamp_size)
    stamp_size = min(max_stamp_size, stamp_size)

    # Make MultiBandObsList
    mb_obs = MultiBandObsList()
    for _, obslist in enumerate(img_mbobs):
        obs_list = ObsList()
        for _, obs in enumerate(obslist):
            img, dx, dy = get_cutout(
                obs.image, det_row["x"], det_row["y"], stamp_size
            )
            wgt, _, _ = get_cutout(
                obs.weight, det_row["x"], det_row["y"], stamp_size
            )
            if hasattr(obs, "noise"):
                noise, _, _ = get_cutout(
                    obs.noise, det_row["x"], det_row["y"], stamp_size
                )
            else:
                noise = None
            if hasattr(obs, "bmask"):
                if np.all(obs.bmask == 0):
                    bmask = np.zeros_like(img, dtype=np.int32)
                else:
                    bmask, _, _ = get_cutout(
                        obs.bmask, det_row["x"], det_row["y"], stamp_size
                    )
            if hasattr(obs, "ormask"):
                if np.all(obs.ormask == 0):
                    ormask = np.zeros_like(img, dtype=np.int32)
                else:
                    ormask, _, _ = get_cutout(
                        obs.ormask, det_row["x"], det_row["y"], stamp_size
                    )
            if do_uberseg:
                seg, _, _ = get_cutout(
                    seg_map, det_row["x"], det_row["y"], stamp_size
                )
                wgt = fast_uberseg(seg, wgt, det_row["number"])

            jac = Jacobian(
                row=dx,
                col=dy,
                dudrow=obs.jacobian.get_dudrow(),
                dudcol=obs.jacobian.get_dudcol(),
                dvdrow=obs.jacobian.get_dvdrow(),
                dvdcol=obs.jacobian.get_dvdcol(),
            )

            newobs = Observation(
                image=img,
                weight=wgt,
                jacobian=jac,
                noise=noise,
                psf=obs.psf,
                bmask=bmask,
                ormask=ormask,
            )

            if hasattr(obs, "ps"):
                newobs.ps = obs.ps

            obs_list.append(newobs)
        mb_obs.append(obs_list)
    return mb_obs


def get_output_cat(n_obj):
    out = np.array(
        list(map(tuple, np.zeros((len(DET_CAT_DTYPE), n_obj)).T)),
        dtype=DET_CAT_DTYPE,
    )
    return out


def get_cat(
    img,
    weight,
    thresh=1.5,
    minarea=5,
    deblend_nthresh=32,
    deblend_cont=0.005,
    kernel=None,
    filter_type="conv",
    header=None,
    wcs=None,
    mask=None,
):
    # NOTE: Might need to look again into this. For now we keep it simple.
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0

    rms = np.median(np.sqrt(1 / weight[m]))
    # rms = mad(img, scale="normal", axis=(0, 1))

    if (header is not None) and (wcs is not None):
        raise ValueError("Only one of header or wcs can be provided.")
    elif header is not None:
        wcs = WCS(header)

    if kernel is None:
        kernel = DES_KERNEL

    # NOTE: Sometimes we end up with a non-zero background, I don't know why..
    bkg = sep.Background(img, mask=mask_rms)

    obj, seg = sep.extract(
        img - bkg.globalback,
        thresh,
        err=rms,
        segmentation_map=True,
        minarea=minarea,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        filter_type=filter_type,
        filter_kernel=np.asarray(kernel),
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

    if wcs is not None:
        ra, dec = wcs.all_pix2world(obj["x"], obj["y"], 0)

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
    out["npix"] = obj["npix"]
    out["x"] = obj["x"]
    out["y"] = obj["y"]
    out["sx_row"] = obj["y"]
    out["sx_col"] = obj["x"]
    out["a"] = obj["a"]
    out["b"] = obj["b"]
    out["xx"] = obj["x2"]
    out["yy"] = obj["y2"]
    out["xy"] = obj["xy"]
    out["elongation"] = obj["a"] / obj["b"]
    out["ellipticity"] = 1.0 - obj["b"] / obj["a"]
    out["kronrad"] = kronrads
    out["flux"] = fluxes
    out["flux_err"] = fluxerrs
    out["flux_radius"] = flux_rad
    out["snr"] = snr
    out["flags"] = obj["flag"]
    out["flux_flags"] = krflags | flags | flags_rad
    out["ext_flags"] = ext_flags
    if wcs is not None:
        out["ra"] = ra
        out["dec"] = dec

    return out, seg


def _shear_positions(x_pix, y_pix, g1, g2, jacobian):
    """Shift pixel positions to account for a metacal shear.

    When metacal applies a reduced shear (g1, g2) to an image, each galaxy
    moves from its noshear pixel position to a new position given by the
    forward action of the shear matrix on the sky-coordinate offset from the
    image centre:

        [u', v'] = S @ [u, v]
        S = [[1+g1,  g2 ],
             [ g2,  1-g1]] / (1 - g1^2 - g2^2)

    This is the reduced-shear convention used by GalSim's ``galsim.Shear``.
    The result is converted back to pixel coordinates via the inverse of the
    Jacobian (WCS).

    Parameters
    ----------
    x_pix, y_pix : array-like
        Truth pixel positions in the noshear image (0-indexed, SEP convention:
        x = col, y = row).
    g1, g2 : float
        Reduced-shear components of the metacal step.
    jacobian : ngmix.Jacobian
        Full-image Jacobian whose ``row0``/``col0`` give the reference pixel
        (image centre) and whose WCS elements map pixel offsets to sky offsets.

    Returns
    -------
    x_new, y_new : ndarray
        Shifted pixel positions (col, row) in the sheared image.
    """
    dudcol = jacobian.get_dudcol()
    dudrow = jacobian.get_dudrow()
    dvdcol = jacobian.get_dvdcol()
    dvdrow = jacobian.get_dvdrow()
    col0 = jacobian.col0
    row0 = jacobian.row0

    # Pixel offsets from image centre → sky offsets (arcsec)
    dcol = np.asarray(x_pix, dtype=np.float64) - col0
    drow = np.asarray(y_pix, dtype=np.float64) - row0
    u = dudcol * dcol + dudrow * drow
    v = dvdcol * dcol + dvdrow * drow

    # Forward reduced-shear transformation (GalSim convention)
    absgsq = g1 * g1 + g2 * g2
    inv_denom = 1.0 / (1.0 - absgsq)
    u_new = inv_denom * ((1.0 + g1) * u + g2 * v)
    v_new = inv_denom * (g2 * u + (1.0 - g1) * v)

    # Sky offsets → pixel offsets via J^{-1}
    det_J = dudcol * dvdrow - dudrow * dvdcol
    dcol_new = (dvdrow * u_new - dudrow * v_new) / det_J
    drow_new = (-dvdcol * u_new + dudcol * v_new) / det_J

    return col0 + dcol_new, row0 + drow_new


def get_cat_force(
    img,
    weight,
    x_pix=None,
    y_pix=None,
    thresh=1.5,
    minarea=5,
    deblend_nthresh=32,
    deblend_cont=0.005,
    kernel=None,
    filter_type="conv",
    header=None,
    wcs=None,
    mask=None,
    g1=0.0,
    g2=0.0,
    jacobian=None,
):
    # NOTE: Might need to look again into this. For now we keep it simple.
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0

    rms = np.median(np.sqrt(1 / weight[m]))
    # rms = mad(img, scale="normal", axis=(0, 1))

    if (header is not None) and (wcs is not None):
        raise ValueError("Only one of header or wcs can be provided.")
    elif header is not None:
        wcs = WCS(header)

    if kernel is None:
        kernel = DES_KERNEL

    if x_pix is not None:
        x = np.asarray(x_pix, dtype=np.float64)
        y = np.asarray(y_pix, dtype=np.float64)
    else:
        x = np.array([img.shape[1] / 2.0])
        y = np.array([img.shape[0] / 2.0])

    # Shift positions to match galaxy locations in the sheared image
    if (g1 != 0.0 or g2 != 0.0) and jacobian is not None:
        x, y = _shear_positions(x, y, g1, g2, jacobian)

    n_obj = len(x)
    seg_id = np.arange(1, n_obj + 1, dtype=np.int32)
    a = np.full(n_obj, 3.0)
    b = np.full(n_obj, 3.0)

    kronrads, krflags = sep.kron_radius(
        img,
        x,
        y,
        a,
        b,
        0,
        6.0,
    )
    fluxes = np.ones(n_obj) * -10.0
    fluxerrs = np.ones(n_obj) * -10.0
    flux_rad = np.ones(n_obj) * -10.0
    snr = np.ones(n_obj) * -10.0
    flags = np.ones(n_obj, dtype=np.int64) * 64
    flags_rad = np.ones(n_obj, dtype=np.int64) * 64

    good_flux = (
        (kronrads > 0)
        & (b > 0)
        & (a >= b)
        & (0 >= -np.pi / 2)
        & (0 <= np.pi / 2)
    )
    fluxes[good_flux], fluxerrs[good_flux], flags[good_flux] = sep.sum_ellipse(
        img,
        x[good_flux],
        y[good_flux],
        a[good_flux],
        b[good_flux],
        0,
        2.5 * kronrads[good_flux],
        err=rms,
        subpix=1,
    )

    flux_rad[good_flux], flags_rad[good_flux] = sep.flux_radius(
        img,
        x[good_flux],
        y[good_flux],
        6.0 * a[good_flux],
        0.5,
        normflux=fluxes[good_flux],
        subpix=1,
    )

    good_snr = (fluxes > 0) & (fluxerrs > 0)
    snr[good_snr] = fluxes[good_snr] / fluxerrs[good_snr]

    if wcs is not None:
        ra, dec = wcs.all_pix2world(x, y, 0)

    out = get_output_cat(n_obj)

    out["number"] = seg_id
    out["x"] = x
    out["y"] = y
    out["sx_row"] = y
    out["sx_col"] = x
    out["a"] = a
    out["b"] = b
    out["xx"] = -1.0
    out["yy"] = -1.0
    out["xy"] = -1.0
    out["elongation"] = a / b
    out["ellipticity"] = 1.0 - b / a
    out["kronrad"] = kronrads
    out["flux"] = fluxes
    out["flux_err"] = fluxerrs
    out["flux_radius"] = flux_rad
    out["snr"] = snr
    out["flags"] = 0
    out["flux_flags"] = krflags | flags | flags_rad
    out["ext_flags"] = 0
    if wcs is not None:
        out["ra"] = ra
        out["dec"] = dec

    return out, None
