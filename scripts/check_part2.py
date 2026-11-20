"""Visual and numerical verification of Part 2 synthetic signal families."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import build_dataset, FS
from src.utils import compute_snr_db

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

c_sin, y_sin, snr_sin = build_dataset(5, "sin_chirp", (5.0, 5.0), seed=100, noise_type="awgn")
c_bpsk, y_bpsk, snr_bpsk = build_dataset(5, "bpsk", (5.0, 5.0), seed=200, noise_type="awgn")

t = np.arange(c_sin.shape[1]) / FS

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t, y_sin[0], color="lightsteelblue", label="Noisy (5 dB AWGN)")
axes[0].plot(t, c_sin[0], color="navy", linewidth=1.3, label="Clean")
axes[0].set_title("Family (a): Random Sinusoids + Linear Chirp")
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, y_bpsk[0], color="lightsalmon", label="Noisy (5 dB AWGN)")
axes[1].plot(t, c_bpsk[0], color="darkred", linewidth=1.3, label="Clean")
axes[1].set_title("Family (b): Pulse-Shaped BPSK Baseband")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = FIG_DIR / "part2_sample_signals.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved sample signals figure -> {out_path}")
print(f"Family (a) measured SNR (target 5.0 dB): {compute_snr_db(c_sin, y_sin):.3f} dB")
print(f"Family (b) measured SNR (target 5.0 dB): {compute_snr_db(c_bpsk, y_bpsk):.3f} dB")