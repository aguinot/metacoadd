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
def zero_pad_fft(im, target_dim):
    padded_im = pad_arr(im, target_dim)
    return fft.rfft2(padded_im)


@nb.njit
def compute_noise_power_spectrum(
    noise_image,
    output_dim,
    get_weights=False,
    normalize=False,
):
    """Compute one-sided (rfft2) noise power spectrum for real-space images.

    Returns a PSD with shape (output_dim, output_dim//2 + 1), normalized
    to mean=1, plus one-sided Hermitian weights for minimal likelihood code.

    Parameters
    ----------
    noise_image : ndarray
        2D array containing pure noise (no signal). Should be square.`
        This is typically obs.noise from ngmix observations.
        Can be larger than the image being measured.
    output_dim : int
        Desired output dimension for the output stamp/grid.
    get_weights : bool, optional
        Whether to return the one-sided Hermitian weights for rfft2 likelihoods.
        Default is False.
    normalize : bool, optional
        If True, normalize PSD by its mean (dimensionless, mean=1).
        If False, keep the absolute noise scale. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        power_spectrum : shape (output_dim, output_dim//2 + 1)
        onesided_weights : shape (output_dim, output_dim//2 + 1) if get_weights else None.
    """
    original_dim = noise_image.shape[0]

    # Compute PSD from original-sized noise, not padded.
    # Padding introduces artificial zeros that would dilute the power spectrum.
    # The PSD should reflect the noise statistics of the actual data only.
    k_noise = fft.rfft2(noise_image)
    # Normalize by original pixel count to preserve absolute scale.
    norm = original_dim * original_dim
    power_spectrum = (
        k_noise.real * k_noise.real + k_noise.imag * k_noise.imag
    ) / norm

    # Resample to output_dim if different from original_dim.
    if output_dim != original_dim:
        power_spectrum = _resample_power_spectrum(
            power_spectrum,
            original_dim,
            output_dim,
            normalize=normalize,
        )
    else:
        if normalize:
            mean_power = np.mean(power_spectrum)
            if mean_power > 0:
                power_spectrum = power_spectrum / mean_power
        # Add floor to avoid exact zeros.
        power_max = np.max(power_spectrum)
        if power_max > 0:
            floor = power_max * 1e-10
            power_spectrum = np.maximum(power_spectrum, floor)

    if get_weights:
        weights = make_rfft2_onesided_weights(output_dim)
        return power_spectrum, weights

    return power_spectrum, None


@nb.njit
def _resample_power_spectrum(
    power_spectrum_rfft,
    input_dim,
    output_dim,
    normalize=False,
):
    """Resample one-sided PSD by resizing correlation then returning one-sided PSD."""
    correlation = fft.irfft2(power_spectrum_rfft, s=(input_dim, input_dim))

    correlation = fft.fftshift(correlation)

    if output_dim > input_dim:
        resampled_correlation = pad_arr(correlation, output_dim)
    else:
        crop_amount = input_dim - output_dim
        crop_before = crop_amount // 2
        crop_after = crop_before + output_dim
        resampled_correlation = correlation[
            crop_before:crop_after,
            crop_before:crop_after,
        ]

    resampled_correlation = fft.ifftshift(resampled_correlation)
    k_resampled = fft.rfft2(resampled_correlation)
    # Correlation -> FFT already returns the PSD. Do not square again.
    # Numerical noise can introduce tiny imaginary parts; keep real part.
    resampled_power_spectrum = k_resampled.real
    resampled_power_spectrum = np.maximum(resampled_power_spectrum, 0.0)

    if normalize:
        mean_power = np.mean(resampled_power_spectrum)
        if mean_power > 0:
            resampled_power_spectrum = resampled_power_spectrum / mean_power

    # Add small floor to prevent exact zeros and near-zero division in chi2/fdiff.
    # Floor is relative to the max power spectrum value to adapt to noise scale.
    power_max = np.max(resampled_power_spectrum)
    if power_max > 0:
        floor = power_max * 1e-10
        resampled_power_spectrum = np.maximum(resampled_power_spectrum, floor)

    return resampled_power_spectrum


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
