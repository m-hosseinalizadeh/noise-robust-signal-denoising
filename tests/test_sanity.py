"""Sanity-check unit tests for metrics, data generation, and classical filters."""

import numpy as np
from src.utils import compute_snr_db, compute_snr_improvement_db
from src.data_gen import build_dataset, add_awgn
from src.classical_filters import (
    moving_average_denoise,
    butterworth_denoise,
    savgol_denoise,
    wiener_denoise,
    estimate_clean_psd,
)


def test_snr_calculation_exact():
    clean = np.ones(1000, dtype=np.float64)
    # Noise with amplitude 0.1 -> power ratio = 1 / 0.01 = 100 -> 20.0 dB
    noisy = clean + 0.1 * np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
    snr = compute_snr_db(clean, noisy)
    assert np.isclose(snr, 20.0, atol=1e-6), f"Expected 20.0 dB, got {snr}"


def test_awgn_achieved_snr():
    rng = np.random.default_rng(42)
    clean = np.sin(2 * np.pi * 50 * np.arange(512) / 1000.0).astype(np.float32)
    for target_snr in [0.0, 10.0, 20.0]:
        noisy = add_awgn(clean, target_snr, rng)
        measured = compute_snr_db(clean, noisy)
        assert np.isclose(measured, target_snr, atol=1e-3), (
            f"Target {target_snr} dB, measured {measured:.4f} dB"
        )


def test_snr_improvement_sign():
    rng = np.random.default_rng(7)
    clean = np.sin(2 * np.pi * 30 * np.arange(512) / 1000.0).astype(np.float32)
    noisy = add_awgn(clean, 5.0, rng)
    better = 0.5 * clean + 0.5 * noisy
    imp = compute_snr_improvement_db(clean, noisy, better)
    assert imp > 0.0, f"Expected positive SNR improvement, got {imp}"


def test_no_leakage_across_seeds_and_unit_rms():
    c1, _, _ = build_dataset(10, "sin_chirp", (0.0, 20.0), seed=101)
    c2, _, _ = build_dataset(10, "sin_chirp", (0.0, 20.0), seed=202)
    assert not np.allclose(c1, c2), "Different seeds must produce distinct datasets"

    for family in ["sin_chirp", "bpsk"]:
        c, y, _ = build_dataset(10, family, (0.0, 20.0), seed=303)
        rms = np.sqrt(np.mean(c ** 2, axis=1))
        assert np.allclose(rms, 1.0, atol=1e-4), f"{family} signals should be unit RMS"
        assert np.all(np.isfinite(y)), f"{family} noisy signals contain NaN/Inf"


def test_classical_filters_preserve_clean_signal():
    """All 4 classical filters should preserve a clean in-band sinusoid with low distortion."""
    t = np.arange(512) / 1000.0
    clean = np.sin(2.0 * np.pi * 40.0 * t).astype(np.float32)

    est_ma = moving_average_denoise(clean, window=3)
    est_bw = butterworth_denoise(clean, cutoff_hz=240.0, order=4)
    est_sg = savgol_denoise(clean, window=7, polyorder=3)
    psd = estimate_clean_psd(clean[None, :])
    est_wi = wiener_denoise(clean, clean_psd=psd, alpha=1.0)

    for name, est in [
        ("moving_average", est_ma),
        ("butterworth", est_bw),
        ("savgol", est_sg),
        ("wiener", est_wi),
    ]:
        snr = compute_snr_db(clean, est)
        assert snr > 20.0, f"{name} distorted clean signal too much (SNR={snr:.2f} dB)"