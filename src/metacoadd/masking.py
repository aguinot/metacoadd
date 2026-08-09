import numpy as np

from metadetect.masking import (
    make_foreground_apodization_mask,
    make_foreground_bmask,
)


def get_radius_arcsec(mag, coeffs):

    ply = np.poly1d(coeffs)
    log10_radius = ply(mag)
    return 10**log10_radius


def get_mask_and_bmask(
    wcs,
    star_cat,
    poly_coeffs,
    mag_max=25,
    min_radius=3,
    ap_radius=7,
    bmask_expand=20,
    bmask_bit_val=8,
):
    """
    Create a mask and a bmask for the foreground stars.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS of the image.
    star_cat : astropy.table.Table
        The catalog of stars with columns 'ra', 'dec', and 'mag_AB'.
    poly_coeffs : list
        The coefficients for the polynomial used to calculate the radius.
    mag_max : float, optional
        The maximum magnitude for the stars to include in the mask.
        Default is 25.
    min_radius : float, optional
        The minimum radius for the star mask in arcsec.
        Default is 3.
    ap_radius : float, optional
        The apodization radius in pixels. Default is 7.
    bmask_expand : float, optional
        The expansion radius for the bmask in pixels. Default is 20.
    bmask_bit_val : int, optional
        The bit value to set in the bmask for the foreground stars.
        Default is 8.

    Returns
    -------
    mask : np.ndarray
        The apodization mask for the foreground stars.
    bmask : np.ndarray
        The bmask for the foreground stars.
    """

    pixel_scale = abs(wcs.pixel_scale_matrix[0][0] * 3600)

    x_stars, y_stars = wcs.all_world2pix(
        star_cat["ra"].values, star_cat["dec"].values, 0
    )

    rad_stars = get_radius_arcsec(star_cat["mag_AB"].values, poly_coeffs)
    rad_stars /= pixel_scale
    rad_stars = np.clip(rad_stars, min_radius, None)

    mag_cut = star_cat["mag_AB"].values < mag_max
    x_stars = x_stars[mag_cut]
    y_stars = y_stars[mag_cut]
    rad_stars = rad_stars[mag_cut]

    mask_ap = make_foreground_apodization_mask(
        xm=x_stars,
        ym=y_stars,
        rm=rad_stars,
        dims=wcs.array_shape,
        symmetrize=False,
        ap_rad=ap_radius,
    )

    bmask = make_foreground_bmask(
        xm=x_stars,
        ym=y_stars,
        rm=rad_stars + bmask_expand,
        dims=wcs.array_shape,
        symmetrize=False,
        mask_bit_val=bmask_bit_val,
    )

    return mask_ap, bmask
