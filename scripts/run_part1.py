"""
Part 1: Sampling, Aliasing, and Uniform Quantization (SQNR Analysis).

Demonstrates:
1. Nyquist-Shannon sampling theorem: multi-tone signal sampled above (1000 Hz)
   and below (300 Hz) the Nyquist rate (2 * 200 Hz = 400 Hz).
2. Uniform mid-tread quantization at 4, 6, 8, and 12 bits, comparing measured
   SQNR against the theoretical 6.02*N + 1.76 dB rule for both a full-scale
   single sinusoid and a multi-tone signal (highlighting the crest-factor offset).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_multitone(t: np.ndarray) -> np.ndarray:
    """Multi-tone signal with components at 50, 120, and 200 Hz (f_max = 200 Hz)."""
    x = (
        0.5 * np.sin(2 * np.pi * 50.0 * t)
        + 0.3 * np.sin(2 * np.pi * 120.0 * t)
        + 0.2 * np.sin(2 * np.pi * 200.0 * t)
    )
    return x


def compute_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum using rFFT."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = (2.0 / n) * np.abs(np.fft.rfft(x))
    mag[0] /= 2.0
    return freqs, mag


def uniform_quantize(x: np.ndarray, n_bits: int, full_scale: float = 1.0) -> np.ndarray:
    """Uniform mid-tread quantizer over [-full_scale, full_scale]."""
    levels = 2 ** n_bits
    step = (2.0 * full_scale) / levels
    x_clipped = np.clip(x, -full_scale, full_scale - step)
    q_idx = np.round(x_clipped / step)
    return q_idx * step


def compute_sqnr_db(clean: np.ndarray, quantized: np.ndarray) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    noise = clean - quantized
    sig_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10.0 * np.log10(sig_power / (noise_power + 1e-15))


def run_aliasing_experiment() -> None:
    duration = 1.0  # seconds

    # Reference "continuous" signal for plotting time-domain overlay
    t_cont = np.linspace(0, 0.05, 5000, endpoint=False)
    x_cont = generate_multitone(t_cont)

    # Case A: Above Nyquist (fs = 1000 Hz > 2 * 200 Hz = 400 Hz)
    fs_high = 1000.0
    t_high = np.arange(0, duration, 1.0 / fs_high)
    x_high = generate_multitone(t_high)
    f_high, mag_high = compute_spectrum(x_high, fs_high)

    # Case B: Below Nyquist (fs = 300 Hz < 400 Hz) -> 200 Hz aliases to |300 - 200| = 100 Hz
    fs_low = 300.0
    t_low = np.arange(0, duration, 1.0 / fs_low)
    x_low = generate_multitone(t_low)
    f_low, mag_low = compute_spectrum(x_low, fs_low)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # Time domain: Above Nyquist
    mask_h = t_high <= 0.05
    axes[0, 0].plot(t_cont * 1e3, x_cont, color="lightgray", linewidth=1.5, label="Continuous ref")
    axes[0, 0].stem(t_high[mask_h] * 1e3, x_high[mask_h], basefmt=" ", label=f"Sampled ({int(fs_high)} Hz)")
    axes[0, 0].set_title(f"Above Nyquist (fs = {int(fs_high)} Hz > 400 Hz)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Frequency domain: Above Nyquist
    axes[0, 1].stem(f_high, mag_high, basefmt=" ")
    axes[0, 1].set_xlim(0, 300)
    axes[0, 1].set_title("Spectrum Above Nyquist (True Peaks: 50, 120, 200 Hz)")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Time domain: Below Nyquist
    mask_l = t_low <= 0.05
    axes[1, 0].plot(t_cont * 1e3, x_cont, color="lightgray", linewidth=1.5, label="Continuous ref")
    axes[1, 0].stem(t_low[mask_l] * 1e3, x_low[mask_l], linefmt="C3-", markerfmt="C3o", basefmt=" ", label=f"Sampled ({int(fs_low)} Hz)")
    axes[1, 0].set_title(f"Below Nyquist (fs = {int(fs_low)} Hz < 400 Hz)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Frequency domain: Below Nyquist
    axes[1, 1].stem(f_low, mag_low, linefmt="C3-", markerfmt="C3o", basefmt=" ")
    axes[1, 1].set_xlim(0, 160)
    axes[1, 1].annotate(
        "Aliased 200 Hz -> 100 Hz",
        xy=(100, 0.2),
        xytext=(55, 0.35),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
        fontweight="bold",
    )
    axes[1, 1].set_title("Spectrum Below Nyquist (200 Hz Folds to 100 Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "part1_aliasing.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved aliasing figure -> {out_path}")


def run_quantization_experiment() -> None:
    fs = 4096.0
    t = np.arange(0, 1.0, 1.0 / fs)

    # 1) Full-scale single sinusoid (prime frequency avoids coherent quantization bias)
    x_sine = np.sin(2 * np.pi * 53.0 * t)

    # 2) Multi-tone signal normalized to peak = 1.0
    x_multi = generate_multitone(t)
    x_multi = x_multi / np.max(np.abs(x_multi))

    bit_depths = [4, 6, 8, 12]
    sqnr_theory = [6.02 * n + 1.76 for n in bit_depths]
    sqnr_sine = []
    sqnr_multi = []

    print("\n--- Part 1: Quantization SQNR Results ---")
    print(f"{'Bits':>4} | {'Theory (dB)':>11} | {'Single Sine (dB)':>16} | {'Multi-tone (dB)':>15}")
    print("-" * 56)
    for n, th in zip(bit_depths, sqnr_theory):
        q_sine = uniform_quantize(x_sine, n, full_scale=1.0)
        q_multi = uniform_quantize(x_multi, n, full_scale=1.0)
        s_sine = compute_sqnr_db(x_sine, q_sine)
        s_multi = compute_sqnr_db(x_multi, q_multi)
        sqnr_sine.append(s_sine)
        sqnr_multi.append(s_multi)
        print(f"{n:>4} | {th:>11.2f} | {s_sine:>16.2f} | {s_multi:>15.2f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bit_depths, sqnr_theory, "k--", linewidth=2, label="Theory: 6.02N + 1.76 dB (full-scale sine)")
    plt.plot(bit_depths, sqnr_sine, "o-", linewidth=2, label="Measured: Single Sinusoid")
    plt.plot(bit_depths, sqnr_multi, "s-", linewidth=2, label="Measured: Multi-tone (higher PAPR)")
    plt.title("Measured vs. Theoretical SQNR Across Bit Depths")
    plt.xlabel("Bit Depth (N bits)")
    plt.ylabel("SQNR (dB)")
    plt.xticks(bit_depths)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "part1_quantization_sqnr.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved SQNR figure -> {out_path}")


if __name__ == "__main__":
    run_aliasing_experiment()
    run_quantization_experiment()