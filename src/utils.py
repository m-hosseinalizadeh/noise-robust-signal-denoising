"""Core reproducibility and signal quality metrics."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_mse(clean: np.ndarray, est: np.ndarray) -> float:
    """Mean Squared Error between clean and estimated signals."""
    return float(np.mean((clean - est) ** 2))


def compute_snr_db(clean: np.ndarray, noisy_or_est: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (dB) for a 1D signal or average over a 2D batch (M, N).
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy_or_est = np.asarray(noisy_or_est, dtype=np.float64)
    err = clean - noisy_or_est
    if clean.ndim == 1:
        sig_p = np.mean(clean ** 2)
        err_p = np.mean(err ** 2)
        return float(10.0 * np.log10((sig_p + 1e-12) / (err_p + 1e-12)))
    sig_p = np.mean(clean ** 2, axis=-1)
    err_p = np.mean(err ** 2, axis=-1)
    return float(np.mean(10.0 * np.log10((sig_p + 1e-12) / (err_p + 1e-12))))


def compute_snr_improvement_db(
    clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray
) -> float:
    """SNR improvement = SNR_out - SNR_in (in dB)."""
    return compute_snr_db(clean, denoised) - compute_snr_db(clean, noisy)