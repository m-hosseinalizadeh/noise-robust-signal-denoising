"""
Part 3: Classical Signal Denoising Baselines and Validation-Only Hyperparameter Tuning.

Methods:
  1. Moving Average (zero-phase symmetric boxcar via uniform_filter1d)
  2. Butterworth Low-Pass (zero-phase IIR via butter + filtfilt)
  3. Savitzky-Golay (local polynomial least-squares smoothing)
  4. Frequency-Domain Wiener Filter (empirical signal PSD + out-of-band noise floor)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, savgol_filter
from src.data_gen import FS
from src.utils import compute_mse, compute_snr_improvement_db


def moving_average_denoise(signals: np.ndarray, window: int) -> np.ndarray:
    """Apply zero-phase moving average along the last axis."""
    return uniform_filter1d(signals, size=window, axis=-1, mode="reflect").astype(np.float32)


def butterworth_denoise(
    signals: np.ndarray, cutoff_hz: float, order: int = 4, fs: float = FS
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter along the last axis."""
    b, a = butter(order, cutoff_hz, btype="low", fs=fs)
    return filtfilt(b, a, signals, axis=-1).astype(np.float32)


def savgol_denoise(signals: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay polynomial filter along the last axis."""
    return savgol_filter(
        signals, window_length=window, polyorder=polyorder, axis=-1, mode="interp"
    ).astype(np.float32)


def estimate_clean_psd(clean_train: np.ndarray) -> np.ndarray:
    """Estimate average one-sided power spectral density |rFFT(x)|^2 from training clean signals."""
    spec = np.fft.rfft(clean_train.astype(np.float64), axis=-1)
    return np.mean(np.abs(spec) ** 2, axis=0)


def wiener_denoise(
    signals: np.ndarray,
    clean_psd: np.ndarray,
    alpha: float = 1.0,
    fs: float = FS,
    noise_band_hz: tuple[float, float] = (350.0, 500.0),
) -> np.ndarray:
    """
    Frequency-domain Wiener filter:
        H(f) = S_xx(f) / (S_xx(f) + alpha * S_nn_hat)
    where S_nn_hat is estimated per signal from the out-of-band high-frequency spectrum.
    """
    x = np.asarray(signals, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]

    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    noise_mask = (freqs >= noise_band_hz[0]) & (freqs <= noise_band_hz[1])

    spec = np.fft.rfft(x, axis=-1)
    # Per-signal white noise PSD estimate from out-of-band bins
    noise_psd = np.mean(np.abs(spec[:, noise_mask]) ** 2, axis=-1, keepdims=True)

    s_xx = clean_psd[None, :]
    gain = s_xx / (s_xx + alpha * noise_psd + 1e-12)
    est = np.fft.irfft(spec * gain, n=n, axis=-1).astype(np.float32)
    return est[0] if squeeze else est


def tune_classical_filters(
    clean_train: np.ndarray, clean_val: np.ndarray, noisy_val: np.ndarray
) -> tuple[dict, np.ndarray]:
    """
    Grid-search hyperparameters for all 4 classical filters strictly on the validation set.
    Returns (best_params_dict, clean_psd_from_train).
    """
    best_params = {}

    # 1. Moving Average
    best_mse = float("inf")
    for w in [3, 5, 7, 9, 11, 15]:
        est = moving_average_denoise(noisy_val, window=w)
        mse = compute_mse(clean_val, est)
        if mse < best_mse:
            best_mse = mse
            best_params["moving_average"] = {"window": w}

    # 2. Butterworth Low-Pass
    best_mse = float("inf")
    for order in [2, 4, 6]:
        for cutoff in [160.0, 190.0, 220.0, 240.0, 260.0, 300.0]:
            est = butterworth_denoise(noisy_val, cutoff_hz=cutoff, order=order)
            mse = compute_mse(clean_val, est)
            if mse < best_mse:
                best_mse = mse
                best_params["butterworth"] = {"cutoff_hz": cutoff, "order": order}

    # 3. Savitzky-Golay
    best_mse = float("inf")
    for w in [5, 7, 9, 11, 15, 21]:
        for p in [2, 3, 4]:
            if p >= w:
                continue
            est = savgol_denoise(noisy_val, window=w, polyorder=p)
            mse = compute_mse(clean_val, est)
            if mse < best_mse:
                best_mse = mse
                best_params["savgol"] = {"window": w, "polyorder": p}

    # 4. Frequency-Domain Wiener
    clean_psd = estimate_clean_psd(clean_train)
    best_mse = float("inf")
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        est = wiener_denoise(noisy_val, clean_psd=clean_psd, alpha=alpha)
        mse = compute_mse(clean_val, est)
        if mse < best_mse:
            best_mse = mse
            best_params["wiener"] = {"alpha": alpha}

    return best_params, clean_psd


def apply_classical_filter(
    name: str, noisy: np.ndarray, params: dict, clean_psd: np.ndarray | None = None
) -> np.ndarray:
    """Dispatch helper to run a named classical filter with its tuned parameters."""
    if name == "moving_average":
        return moving_average_denoise(noisy, **params)
    if name == "butterworth":
        return butterworth_denoise(noisy, **params)
    if name == "savgol":
        return savgol_denoise(noisy, **params)
    if name == "wiener":
        if clean_psd is None:
            raise ValueError("clean_psd is required for wiener filter")
        return wiener_denoise(noisy, clean_psd=clean_psd, **params)
    raise ValueError(f"Unknown classical filter: {name}")