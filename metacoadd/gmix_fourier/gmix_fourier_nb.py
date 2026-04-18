"""
Analytic Fourier-space evaluation of a Gaussian mixture model.

For a convolved gmix (galaxy * PSF) where every component has moments
(irr, irc, icc) large enough for the Fourier-space Gaussian to decay
meaningfully (guaranteed once the PSF moments are added), we can
compute the rfft2 of the model *without any FFT*, using:

    k_model[fr, fc] = sum_i  p_i
                        * exp(-2*pi^2*(irr_i*fr^2 + 2*irc_i*fr*fc + icc_i*fc^2))
                        * exp(-2*pi*i*(fr*(row0 + row_i) + fc*(col0 + col_i)))

where (fr, fc) are cycles-per-pixel frequencies from fftfreq / rfftfreq,
(row0, col0) is the Jacobian origin in pixel coordinates, and (row_i, col_i)
are the Gaussian centroids in sky/Jacobian coordinates (same as ngmix's
gauss['row'], gauss['col']).

The phase decomposition

    exp(-2*pi*i*(fr*(row0+row_i) + fc*(col0+col_i)))
    = exp(-2*pi*i*fr*(row0+row_i))   [1-D vector over rows]
    * exp(-2*pi*i*fc*(col0+col_i))   [1-D vector over cols]

is exploited for efficiency: we compute the two 1-D complex exponentials
first and form their outer product, avoiding a full N×(N//2+1) complex
exp per component.

Two public entry points are provided:

* ``gmix_eval_fourier_analytic(gmix, N, row0, col0)``
    Evaluate on a fresh output array.  Call once per function evaluation.

* ``gmix_eval_fourier_analytic_inplace(gmix, N, row0, col0, out)``
    Fill a pre-allocated (N, N//2+1) complex array.  Avoids allocation
    in the hot loop; pass ``out`` with ``out[:] = 0`` before calling.

Both are @nb.njit and safe to call from other jitted functions.
"""

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _eval_single_gauss_fourier(
    p,
    row_g,  # sky-coord centroid (v), i.e. gauss["row"]
    col_g,  # sky-coord centroid (u), i.e. gauss["col"]
    irr,  # sky-coord moment (arcsec^2)
    irc,
    icc,
    fr,  # 1-D fftfreq array  (length N), cycles/pixel
    fc,  # 1-D rfftfreq array (length N//2+1), cycles/pixel
    row0,  # Jacobian row origin in pixel coords
    col0,  # Jacobian col origin in pixel coords
    dvdrow,  # Jacobian elements: [v, u] = J * [delta_row, delta_col]
    dvdcol,
    dudrow,
    dudcol,
    out,  # (N, N//2+1) complex128 accumulator
):
    """Accumulate one Gaussian component into *out* (in-place, no alloc).

    The gmix stores moments and centroids in sky (arcsec) coordinates.
    The DFT operates on pixel indices, so we convert via the Jacobian:

        J = [[dvdrow, dvdcol],      maps pixel offsets -> sky offsets
             [dudrow, dudcol]]

        J^{-1} maps sky -> pixel offsets.

    Pixel-space moments:
        C_pix = J^{-1} * C_sky * (J^{-1})^T

    Pixel-space centroid offset from (row0, col0):
        [dr_g, dc_g] = J^{-1} * [v_g, u_g]
    """
    N = fr.shape[0]
    Nc = fc.shape[0]

    two_pi = 2.0 * np.pi
    two_pi2 = 2.0 * np.pi * np.pi

    # ------------------------------------------------------------------
    # Jacobian inverse: J^{-1} = [[dudcol, -dvdcol], [-dudrow, dvdrow]] / det_J
    # ------------------------------------------------------------------
    det_J = dvdrow * dudcol - dvdcol * dudrow
    inv_det = 1.0 / det_J
    A = dudcol * inv_det  # J^{-1}[0, 0]
    B = -dvdcol * inv_det  # J^{-1}[0, 1]
    C = -dudrow * inv_det  # J^{-1}[1, 0]
    D = dvdrow * inv_det  # J^{-1}[1, 1]

    # ------------------------------------------------------------------
    # Pixel-space centroid (absolute pixel position)
    # ------------------------------------------------------------------
    dr_g = A * row_g + B * col_g  # pixel offset from (row0, col0)
    dc_g = C * row_g + D * col_g
    row_pix = row0 + dr_g
    col_pix = col0 + dc_g

    # ------------------------------------------------------------------
    # Pixel-space moments: C_pix = J^{-1} * C_sky * (J^{-1})^T
    #   irr_pix = A^2*irr + 2*A*B*irc + B^2*icc
    #   irc_pix = A*C*irr + (A*D + B*C)*irc + B*D*icc
    #   icc_pix = C^2*irr + 2*C*D*irc + D^2*icc
    # ------------------------------------------------------------------
    irr_pix = A * A * irr + 2.0 * A * B * irc + B * B * icc
    irc_pix = A * C * irr + (A * D + B * C) * irc + B * D * icc
    icc_pix = C * C * irr + 2.0 * C * D * irc + D * D * icc

    # ------------------------------------------------------------------
    # 1-D phase vectors (outer-product trick: N + N//2+1 trig calls
    # instead of N*(N//2+1) per component)
    # ------------------------------------------------------------------
    phase_r_re = np.empty(N)
    phase_r_im = np.empty(N)
    for i in range(N):
        ang = -two_pi * fr[i] * row_pix
        phase_r_re[i] = np.cos(ang)
        phase_r_im[i] = np.sin(ang)

    phase_c_re = np.empty(Nc)
    phase_c_im = np.empty(Nc)
    for j in range(Nc):
        ang = -two_pi * fc[j] * col_pix
        phase_c_re[j] = np.cos(ang)
        phase_c_im[j] = np.sin(ang)

    # ------------------------------------------------------------------
    # Accumulate over the frequency grid
    # ------------------------------------------------------------------
    for i in range(N):
        fri = fr[i]
        irr_fri2 = irr_pix * fri * fri
        two_irc_fri = 2.0 * irc_pix * fri

        pr_re = phase_r_re[i]
        pr_im = phase_r_im[i]

        for j in range(Nc):
            fcj = fc[j]
            exponent = two_pi2 * (
                irr_fri2 + two_irc_fri * fcj + icc_pix * fcj * fcj
            )
            envelope = p * np.exp(-exponent)

            pc_re = phase_c_re[j]
            pc_im = phase_c_im[j]
            full_re = pr_re * pc_re - pr_im * pc_im
            full_im = pr_re * pc_im + pr_im * pc_re

            out[i, j] = out[i, j] + nb.complex128(
                complex(envelope * full_re, envelope * full_im)
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def gmix_eval_fourier_analytic_inplace(
    gmix, N, row0, col0, dvdrow, dvdcol, dudrow, dudcol, out
):
    """Evaluate gmix in Fourier space, accumulating into *out*.

    Parameters
    ----------
    gmix : structured ndarray
        Gaussian mixture data array as returned by ``GMix.get_data()``.
        Must have fields: ``p``, ``row``, ``col``, ``irr``, ``irc``, ``icc``.
        Moments and centroids are in sky (arcsec) coordinates.
    N : int
        Stamp size (rfft2 rows).
    row0, col0 : float
        Jacobian origin in pixel coordinates.
    dvdrow, dvdcol, dudrow, dudcol : float
        Elements of the 2x2 Jacobian matrix J that maps pixel offsets to sky
        offsets: [v, u] = J * [delta_row, delta_col].  For a diagonal
        Jacobian with pixel scale h: dvdrow=h, dudcol=h, dvdcol=dudrow=0.
    out : ndarray, complex128, shape (N, N//2+1)
        Pre-allocated output array.  Caller is responsible for zeroing before
        calling.
    """
    # Frequency axes (cycles / pixel)
    fr = np.empty(N)
    for i in range(N):
        k = i if i < (N + 1) // 2 else i - N
        fr[i] = k / N

    Nc = N // 2 + 1
    fc = np.empty(Nc)
    for j in range(Nc):
        fc[j] = j / N

    n_gauss = gmix.shape[0]
    for ig in range(n_gauss):
        g = gmix[ig]
        _eval_single_gauss_fourier(
            g["p"],
            g["row"],
            g["col"],
            g["irr"],
            g["irc"],
            g["icc"],
            fr,
            fc,
            row0,
            col0,
            dvdrow,
            dvdcol,
            dudrow,
            dudcol,
            out,
        )


@nb.njit(cache=True)
def gmix_eval_fourier_analytic(
    gmix, N, row0, col0, dvdrow, dvdcol, dudrow, dudcol
):
    """Evaluate gmix in Fourier space, returning a new array.

    Parameters
    ----------
    gmix : structured ndarray
        Gaussian mixture data array (``GMix.get_data()``).
        Moments and centroids are in sky (arcsec) coordinates.
    N : int
        Stamp size (rfft2 rows).
    row0, col0 : float
        Jacobian origin in pixel coordinates.
    dvdrow, dvdcol, dudrow, dudcol : float
        Jacobian matrix elements: [v, u] = J * [delta_row, delta_col].

    Returns
    -------
    out : ndarray, complex128, shape (N, N//2+1)
    """
    Nc = N // 2 + 1
    out = np.zeros((N, Nc), dtype=nb.complex128)
    gmix_eval_fourier_analytic_inplace(
        gmix, N, row0, col0, dvdrow, dvdcol, dudrow, dudcol, out
    )
    return out
