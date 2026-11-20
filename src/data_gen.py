"""
Part 2 & Part 5: Synthetic 1D Signal Families and Noise Models.

Signal families (length N=512, fs=1000 Hz, normalized to unit RMS power):
  - 'sin_chirp': Sum of 2-4 random sinusoids (20-180 Hz) + linear chirp.
  - 'bpsk':      Random +-1 BPSK symbols shaped with a Raised-Cosine pulse.

Noise models (scaled to exact target SNR in dB):
  - 'awgn':      Additive White Gaussian Noise (used in training).
  - 'colored':   1/f^alpha colored Gaussian noise (OOD test in Part 5).
  - 'impulsive': Gaussian background + sparse heavy impulses (OOD test in Part 5).
"""

import numpy as np
from scipy.signal import chirp


FS = 1000.0
SIGNAL_LEN = 512


def _normalize_rms(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    return (x / rms).astype(np.float32)


def generate_sinusoids_plus_chirp(
    rng: np.random.Generator, n_samples: int = SIGNAL_LEN, fs: float = FS
) -> np.ndarray:
    """Family (a): Sum of random sinusoids plus a linear chirp."""
    t = np.arange(n_samples) / fs
    x = np.zeros(n_samples, dtype=np.float64)

    n_tones = int(rng.integers(2, 5))
    freqs = rng.uniform(20.0, 180.0, size=n_tones)
    amps = rng.uniform(0.3, 1.0, size=n_tones)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_tones)
    for f, a, ph in zip(freqs, amps, phases):
        x += a * np.sin(2.0 * np.pi * f * t + ph)

    f0 = float(rng.uniform(25.0, 80.0))
    f1 = float(rng.uniform(120.0, 220.0))
    chirp_amp = float(rng.uniform(0.4, 0.9))
    x += chirp_amp * chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear")

    return _normalize_rms(x)


def _raised_cosine_filter(sps: int, beta: float = 0.35, span: int = 6) -> np.ndarray:
    """Truncated Raised-Cosine pulse-shaping filter."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    t = n / float(sps)
    h = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if np.isclose(abs(2.0 * beta * ti), 1.0):
            h[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
        else:
            h[i] = np.sinc(ti) * np.cos(np.pi * beta * ti) / (1.0 - (2.0 * beta * ti) ** 2)
    return h / np.sqrt(np.sum(h ** 2))


def generate_bpsk_baseband(
    rng: np.random.Generator,
    n_samples: int = SIGNAL_LEN,
    sps: int = 16,
    beta: float = 0.35,
) -> np.ndarray:
    """Family (b): Pulse-shaped BPSK baseband signal."""
    n_symbols = int(np.ceil(n_samples / sps)) + 8
    bits = rng.choice([-1.0, 1.0], size=n_symbols)
    upsampled = np.zeros(n_symbols * sps, dtype=np.float64)
    upsampled[::sps] = bits

    rc_filter = _raised_cosine_filter(sps=sps, beta=beta, span=6)
    shaped = np.convolve(upsampled, rc_filter, mode="same")

    # Crop central segment to avoid filter edge transients
    start = (len(shaped) - n_samples) // 2
    x = shaped[start : start + n_samples]
    return _normalize_rms(x)


def _scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(clean.astype(np.float64) ** 2)
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    cur_noise_power = np.mean(noise ** 2) + 1e-12
    scaled_noise = noise * np.sqrt(target_noise_power / cur_noise_power)
    return (clean + scaled_noise).astype(np.float32)


def add_awgn(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add White Gaussian Noise at exact target SNR (dB)."""
    noise = rng.standard_normal(size=clean.shape)
    return _scale_noise_to_snr(clean, noise, snr_db)


def add_colored_noise(
    clean: np.ndarray, snr_db: float, rng: np.random.Generator, alpha: float = 1.5
) -> np.ndarray:
    """Add 1/f^alpha colored Gaussian noise (low-frequency heavy) at exact SNR (dB)."""
    white = rng.standard_normal(size=clean.shape)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(clean), d=1.0 / FS)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))
    scale[0] = 0.0
    colored = np.fft.irfft(spec * scale, n=len(clean))
    return _scale_noise_to_snr(clean, colored, snr_db)


def add_impulsive_noise(
    clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    p_impulse: float = 0.02,
    impulse_std: float = 10.0,
) -> np.ndarray:
    """Add Gaussian background + sparse heavy impulses (Bernoulli-Gaussian) at exact SNR (dB)."""
    n = len(clean)
    bg = rng.standard_normal(n)
    mask = rng.random(n) < p_impulse
    impulses = mask * rng.normal(0.0, impulse_std, size=n)
    return _scale_noise_to_snr(clean, bg + impulses, snr_db)


def build_dataset(
    n_signals: int,
    family: str,
    snr_range_db: tuple[float, float],
    seed: int,
    noise_type: str = "awgn",
    n_samples: int = SIGNAL_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a reproducible dataset (clean, noisy, snrs) using an isolated RNG seed.
    Using distinct seeds for train, val, and test guarantees zero data leakage.
    """
    rng = np.random.default_rng(seed)
    clean_arr = np.zeros((n_signals, n_samples), dtype=np.float32)
    noisy_arr = np.zeros((n_signals, n_samples), dtype=np.float32)
    snrs_arr = rng.uniform(snr_range_db[0], snr_range_db[1], size=n_signals).astype(np.float32)

    for i in range(n_signals):
        if family == "sin_chirp":
            x = generate_sinusoids_plus_chirp(rng, n_samples=n_samples)
        elif family == "bpsk":
            x = generate_bpsk_baseband(rng, n_samples=n_samples)
        else:
            raise ValueError(f"Unknown signal family: {family}")

        if noise_type == "awgn":
            y = add_awgn(x, float(snrs_arr[i]), rng)
        elif noise_type == "colored":
            y = add_colored_noise(x, float(snrs_arr[i]), rng)
        elif noise_type == "impulsive":
            y = add_impulsive_noise(x, float(snrs_arr[i]), rng)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        clean_arr[i] = x
        noisy_arr[i] = y

    return clean_arr, noisy_arr, snrs_arr