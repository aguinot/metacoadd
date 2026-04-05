"""
The code in this file is mainly AI generated using Claude Opus 4.6
It has been tested and validated.
"""

import numba as nb

import numpy as np
from numpy import fft


@nb.njit
def pad_arr(arr, target_dim):

    pad_width = (target_dim - arr.shape[0]) // 2
    padded_arr = np.zeros((target_dim, target_dim))
    end = pad_width + arr.shape[0]
    padded_arr[pad_width:end, pad_width:end] = arr
    return padded_arr


@nb.njit
def meshgrid_2d(x, y):
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for i, y_ in enumerate(y):
        for j, x_ in enumerate(x):
            xx[i, j] = x_
            yy[i, j] = y_
    return xx, yy


@nb.njit
def zero_pad_fft(im, target_dim):
    padded_im = pad_arr(im, target_dim)
    return fft.rfft2(padded_im)


@nb.njit
def compute_noise_power_spectrum(
    noise_image,
):
    """Compute one-sided (rfft2) noise power spectrum from a single noise image.

    Parameters
    ----------
    noise_image : ndarray (N, N)
        Square 2-D array containing pure noise (should be mean-subtracted).

    Returns
    -------
    power_spectrum : ndarray (N, N // 2 + 1)
    """
    N = noise_image.shape[0]
    k_noise = fft.rfft2(noise_image)
    norm = N * N
    ps = (k_noise.real * k_noise.real + k_noise.imag * k_noise.imag) / norm

    # Floor to prevent exact zeros / near-zero division in chi2
    power_max = np.max(ps)
    if power_max > 0:
        ps = np.maximum(ps, power_max * 1e-10)

    return ps


@nb.njit
def estimate_noise_ps_analytic(
    noise_image,
    stamp_size,
    target_variance=0.0,
    correct_periodicity=True,
):
    """Estimate the stamp-level noise PSD analytically (mimics GalSim).

    Reproduces GalSim's ``CorrelatedNoise`` pipeline without GalSim:

    1.  Compute the correlation function (CF) from the template
        periodogram.
    2.  Apply the periodicity-dilution correction (same as GalSim's
        ``correct_periodicity=True``).
    3.  Truncate the CF to stamp-sized lags (what GalSim's
        ``InterpolatedImage.drawImage`` does when the pixel scales
        match).
    4.  FFT the truncated CF to obtain the stamp PSD.

    If several independent noise realisations of the same field are
    available, pass them as a list to ``noise_images``: their
    periodograms are averaged before step 2, giving a lower-variance
    CF (and therefore a better PSD).

    Parameters
    ----------
    noise_images : ndarray (L, L)
        One or more square noise template images
    stamp_size : int
        Side length of the fitting stamps.
    target_variance : float, optional
        If > 0, rescale the PSD so that its implied pixel variance
        equals this value.
    correct_periodicity : bool, optional
        Apply GalSim's periodicity-dilution correction to the CF
        (default ``True``).

    Returns
    -------
    ps : ndarray (stamp_size, stamp_size // 2 + 1)
        One-sided rfft2 power spectrum at stamp resolution.
    """

    L = noise_image.shape[0]
    N = stamp_size
    if L < N:
        raise ValueError(f"noise_image size ({L}) must be >= stamp_size ({N})")

    # ------------------------------------------------------------------
    # Step 1 – average periodogram over all supplied templates
    # ------------------------------------------------------------------
    ps_full = np.zeros((L, L // 2 + 1), dtype=np.float64)
    ft = fft.rfft2(noise_image)
    ps_full += ft.real**2 + ft.imag**2
    ps_full /= (len(noise_image) + 1) * (L * L)

    # ------------------------------------------------------------------
    # Step 2 – CF from periodogram, with periodicity correction
    # ------------------------------------------------------------------
    cf = fft.irfft2(ps_full, s=(L, L))

    if correct_periodicity:
        # GalSim's _cf_periodicity_dilution_correction
        delta_x = fft.fftfreq(L) * L
        delta_y = fft.fftfreq(L) * L
        # Dx, Dy = np.meshgrid(delta_x, delta_y)
        Dx, Dy = meshgrid_2d(delta_x, delta_y)
        correction = (L * L) / ((L - np.abs(Dx)) * (L - np.abs(Dy)))
        cf *= correction

    # ------------------------------------------------------------------
    # Step 3 – truncate CF to stamp-sized lags (FFT order)
    # ------------------------------------------------------------------
    half = N // 2
    n_pos = half + 1  # lags 0 .. half
    n_neg = N - n_pos  # lags -(N-n_pos) .. -1

    cf_stamp = np.zeros((N, N), dtype=np.float64)
    cf_stamp[:n_pos, :n_pos] = cf[:n_pos, :n_pos]
    cf_stamp[:n_pos, n_pos:] = cf[:n_pos, L - n_neg :]
    cf_stamp[n_pos:, :n_pos] = cf[L - n_neg :, :n_pos]
    cf_stamp[n_pos:, n_pos:] = cf[L - n_neg :, L - n_neg :]

    # ------------------------------------------------------------------
    # Step 4 – FFT → stamp PSD
    # ------------------------------------------------------------------
    ps_complex = fft.rfft2(cf_stamp)
    # For a valid positive-definite CF, this should be real & positive;
    # we take abs() as GalSim does to handle minor numerical noise.
    ps = np.abs(ps_complex)

    # ------------------------------------------------------------------
    # Rescale to target variance
    # ------------------------------------------------------------------
    if target_variance > 0.0:
        w = make_rfft2_onesided_weights(N)
        current_var = np.sum(w * ps) / (N * N)
        if current_var > 0.0:
            ps *= target_variance / current_var

    # Floor
    ps_max = np.max(ps)
    if ps_max > 0:
        ps = np.maximum(ps, ps_max * 1e-12)

    return ps


@nb.njit
def make_rfft2_onesided_weights(dim):
    """Hermitian multiplicity weights for one-sided rfft2 likelihood sums."""
    ncol = dim // 2 + 1
    weights = np.full((dim, ncol), 2.0)
    weights[:, 0] = 1.0
    if dim % 2 == 0:
        weights[:, ncol - 1] = 1.0
    return weights


@nb.njit
def chisq_from_rfft2_residual(
    data_k,
    model_k,
    noise_power_rfft,
    onesided_weights,
):
    """Compute chi2 and s2n terms from one-sided Fourier quantities.

    Returns
    -------
    tuple(float, float, float)
        chi2, s2n_numer, s2n_denom
    """

    # Parseval factor for numpy FFT conventions (forward unnormalized).
    nrow = data_k.shape[0]
    norm = 1.0 / (nrow * nrow)

    inv_noise = norm * onesided_weights / noise_power_rfft

    residual_k = data_k - model_k

    chi2 = np.sum(
        inv_noise
        * (
            residual_k.real * residual_k.real
            + residual_k.imag * residual_k.imag
        )
    )

    s2n_numer = np.sum(
        inv_noise * (data_k.real * model_k.real + data_k.imag * model_k.imag)
    )

    s2n_denom = np.sum(
        inv_noise * (model_k.real * model_k.real + model_k.imag * model_k.imag)
    )

    return chi2, s2n_numer, s2n_denom


@nb.njit
def fill_fdiff_from_rfft2(
    data_k,
    model_k,
    noise_power_rfft,
    onesided_weights,
    fdiff,
    start,
):
    """Fill fdiff with weighted one-sided Fourier residuals.

    This is the Fourier analog of (model - data) * ierr used in ngmix.
    Real and imaginary parts are appended separately so that
    sum(fdiff**2) is the Fourier chi2.
    """
    nrow, ncol = data_k.shape
    index = start

    # Keep fdiff consistent with chi2 in real space via Parseval scaling.
    norm_sqrt = 1.0 / nrow

    for i in range(nrow):
        for j in range(ncol):
            diff = model_k[i, j] - data_k[i, j]
            w = norm_sqrt * np.sqrt(
                onesided_weights[i, j] / noise_power_rfft[i, j]
            )

            fdiff[index] = diff.real * w
            index += 1
            fdiff[index] = diff.imag * w
            index += 1
