import copy

from math import sqrt

import numpy as np
import numba as nb

import galsim

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
    """Compute the size of the cutout based on the second moments of the object.

    Parameters
    ----------
    Qxx, Qxy, Qyy : float
        Second moments of the object.
    n_sigma : float
        Number of sigma to include in the cutout size.

    Returns
    -------
    cutout_size : float
        Size of the cutout in pixels.

    """
    # Compute trace and determinant
    trace = Qxx + Qyy

    # Compute eigenvalues analytically for symmetric 2x2 matrix
    temp = sqrt((Qxx - Qyy) ** 2 + 4 * Qxy**2)
    lam1 = 0.5 * (trace + temp)
    lam2 = 0.5 * (trace - temp)

    lam_max = max(lam1, lam2)

    return 2.0 * n_sigma * sqrt(lam_max)


def get_cutout(img, x, y, stamp_size):
    """Get a cutout from the image centered at (x, y) with size stamp_size.
    Also returns the coordinates of the center in the cutout frame.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    x, y : float
        Center of the cutout in pixel coordinates (0-indexed).
    stamp_size : int
        Size of the cutout (must be odd).

    Returns
    -------
    cutout : np.ndarray
        The cutout image.
    cutout_row : float
        The row coordinate of the center in the cutout frame.
    cutout_col : float
        The column coordinate of the center in the cutout frame.

    """
    orow = int(y)
    ocol = int(x)
    half_box_size = stamp_size // 2
    maxrow, maxcol = img.shape

    # Requested bounds before clipping (keep your current centering convention)
    req_row0 = orow - half_box_size + 1
    req_col0 = ocol - half_box_size + 1
    req_row1 = req_row0 + stamp_size
    req_col1 = req_col0 + stamp_size

    # Overlap with image
    src_row0 = max(0, req_row0)
    src_col0 = max(0, req_col0)
    src_row1 = min(maxrow, req_row1)
    src_col1 = min(maxcol, req_col1)

    # Fixed-size zero-padded output
    cutout = np.zeros((stamp_size, stamp_size), dtype=img.dtype)

    if (src_row1 > src_row0) and (src_col1 > src_col0):
        dst_row0 = src_row0 - req_row0
        dst_col0 = src_col0 - req_col0
        dst_row1 = dst_row0 + (src_row1 - src_row0)
        dst_col1 = dst_col0 + (src_col1 - src_col0)

        cutout[dst_row0:dst_row1, dst_col0:dst_col1] = img[
            src_row0:src_row1, src_col0:src_col1
        ]

    # Coordinates in the padded cutout frame
    cutout_row = y - req_row0
    cutout_col = x - req_col0

    return cutout, cutout_row, cutout_col


def get_stamp_mbobs(
    img_mbobs,
    det_row,
    min_stamp_size=71,
    max_stamp_size=201,
    do_uberseg=False,
    seg_map=None,
):
    """Get a MultiBandObsList of cutouts centered on the detected object.

    Parameters
    ----------
    img_mbobs : ngmix.MultiBandObsList
        Multi-band observations of the image.
    det_row : np.ndarray
        Row of the detection catalog corresponding to the object.
    min_stamp_size : int
        Minimum size of the stamp (must be odd).
    max_stamp_size : int
        Maximum size of the stamp (must be odd).
    do_uberseg : bool, optional
        Whether to apply uberseg to the weight map using the segmentation map.
        Default is False.
    seg_map : np.ndarray, optional
        Segmentation map of the image. Required if do_uberseg is True.
        Default is None.

    Returns
    -------
    mb_obs : ngmix.MultiBandObsList
        Multi-band observations of the cutouts.

    """
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

            # Temporary fix to avoid empty weight errors
            if np.all(wgt == 0):
                wgt[0, 0] = 1.0  # Avoid empty weight map

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
    """Get an empty output catalog with the correct dtype.

    Parameters
    ----------
    n_obj : int
        Number of objects in the catalog.

    Returns
    -------
    out : np.ndarray
        Empty output catalog with the correct dtype.

    """
    out = np.array(
        list(map(tuple, np.zeros((len(DET_CAT_DTYPE), n_obj)).T)),
        dtype=DET_CAT_DTYPE,
    )
    return out


def get_pixel_scale(wcs):
    """Get the pixel scale from a WCS object.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or galsim.wcs.BaseWCS
        The WCS object from which to extract the pixel scale.

    Returns
    -------
    pixel_scale : float
        The pixel scale in arcsec/pixel.

    """
    if isinstance(wcs, WCS):
        mat = wcs.pixel_scale_matrix
        pixel_scale = mat[0, 0] * 3600.0  # arcsec/pixel
    elif isinstance(wcs, galsim.wcs.BaseWCS):
        pixel_scale = np.sqrt(wcs.pixelArea(wcs.origin))
    elif wcs is None:
        pixel_scale = 1.0
    else:
        raise ValueError(
            "wcs must be an astropy.wcs.WCS or galsim.wcs.BaseWCS object"
        )
    return pixel_scale


def get_filter_kernel(kernel, wcs=None):
    """Get the filter kernel for SEP.

    Parameters
    ----------
    kernel : list or dict
        The kernel to use for filtering. If a list, it is assumed to be a 2D
        array. If a dict, it is assumed to be a galsim config dict.
    wcs : astropy.wcs.WCS or galsim.wcs.BaseWCS, optional
        The WCS object to use for the kernel. Required if kernel is a dict.
        Default is None.

    Returns
    -------
    filter_kernel : np.ndarray
        The filter kernel as a 2D array.

    """
    if isinstance(kernel, list):
        kernel = np.asarray(kernel)
    elif isinstance(kernel, dict):
        pixel_scale = get_pixel_scale(wcs)
        obj, _ = galsim.config.BuildGSObject({"": kernel}, "")
        if obj is None:
            raise ValueError("Failed to build kernel from config.")
        kernel = obj.drawImage(scale=pixel_scale).array
    else:
        raise ValueError(f"kernel must be a list or a dict, got {type(kernel)}")
    return kernel


def get_xyToradec_func(wcs):
    """Get a function that converts pixel coordinates to world coordinates based
    on the type of WCS object.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or galsim.wcs.BaseWCS
        The WCS object from which to extract the conversion function.

    Returns
    -------
    xyToradec : function
        A function that takes x, y pixel coordinates and returns ra, dec.

    """
    if isinstance(wcs, WCS):

        def xyToradec(x, y):
            return wcs.all_pix2world(x, y, 0)
    elif isinstance(wcs, galsim.wcs.BaseWCS):

        def xyToradec(x, y):
            if wcs.isCelestial():
                return wcs.xyToradec(x, y, units=galsim.degrees)
            else:
                return wcs.xyTouv(x, y)
    else:
        raise ValueError(
            "wcs must be an astropy.wcs.WCS or galsim.wcs.BaseWCS object"
        )

    return xyToradec


def get_cat(
    img,
    weight,
    thresh_type="relative",
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
    """Get a catalog of detected objects from an image using SEP.

    Parameters
    ----------
    img : np.ndarray
        The input image.
    weight : np.ndarray
        The weight image.
    thresh_type : str, optional
        The type of threshold to use in ["relative", "absolute"].
        Default is "relative".
    thresh : float, optional
        The threshold value. Default is 1.5.
    minarea : int, optional
        The minimum number of pixels of a detected object. Default is 5.
    deblend_nthresh : int, optional
        The number of thresholds to use for deblending. Default is 32.
    deblend_cont : float, optional
        The contrast threshold for deblending. Default is 0.005.
    kernel : array-like or dict, optional
        The kernel to use for filtering. Can be a 2D array or a galsim config
        dict. Default is None, which uses the DES kernel.
    filter_type : str, optional
        The type of filter to use in ["conv", "match"]. Default is "conv".
    header : astropy.io.fits.Header, optional
        The header of the input image. Default is None.
    wcs : astropy.wcs.WCS or galsim.wcs.BaseWCS, optional
        The WCS object of the input image. Default is None.
    mask : np.ndarray, optional
        A mask to apply to the input image. Default is None.

    Returns
    -------
    catalog : np.ndarray
        A table containing the detected objects.

    """
    # NOTE: Might need to look again into this. For now we keep it simple.
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0

    # rms = np.median(np.sqrt(1 / weight[m]))
    # rms = mad(img, scale="normal", axis=(0, 1))

    if (header is not None) and (wcs is not None):
        raise ValueError("Only one of header or wcs can be provided.")
    elif header is not None:
        wcs = WCS(header)

    # if kernel is None:
    #     kernel = DES_KERNEL
    filter_kernel = get_filter_kernel(kernel, wcs=wcs)

    # NOTE: Sometimes we end up with a non-zero background, I don't know why..
    # bkg = sep.Background(img, mask=mask_rms)

    if thresh_type == "relative":
        detect_err = rms
    elif thresh_type == "absolute":
        detect_err = None
    else:
        raise ValueError(
            f"Unknown thresh_type: {thresh_type}. "
            "Must be one of ['relative', 'absolute']."
        )
    obj, seg = sep.extract(
        img,  # - bkg.globalback,
        thresh,
        err=detect_err,
        segmentation_map=True,
        minarea=minarea,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        filter_type=filter_type,
        filter_kernel=filter_kernel,
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

    # Disable for now as we don't pass the WCS
    # if wcs is not None:
    #     xyToradec = get_xyToradec_func(wcs)
    #     ra, dec = xyToradec(obj["x"], obj["y"])

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
    # if wcs is not None:
    #     out["ra"] = ra
    #     out["dec"] = dec

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
