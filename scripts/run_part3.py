"""
Part 3: Tune classical filters strictly on the validation set and save tuned parameters.
"""

import json
from pathlib import Path
import numpy as np
from src.data_gen import build_dataset
from src.classical_filters import tune_classical_filters, apply_classical_filter
from src.utils import compute_mse, compute_snr_improvement_db

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Distinct seeds for train and val to prevent any leakage
    clean_train, _, _ = build_dataset(
        n_signals=2000, family="sin_chirp", snr_range_db=(0.0, 20.0), seed=1001, noise_type="awgn"
    )
    clean_val, noisy_val, _ = build_dataset(
        n_signals=400, family="sin_chirp", snr_range_db=(0.0, 20.0), seed=2001, noise_type="awgn"
    )

    best_params, clean_psd = tune_classical_filters(clean_train, clean_val, noisy_val)

    print("\n--- Part 3: Validation-Tuned Classical Baselines (sin_chirp, AWGN 0-20 dB) ---")
    print(f"{'Method':<16} | {'Val MSE':>9} | {'Val SNR Imp (dB)':>16} | Best Params")
    print("-" * 76)
    for name in ["moving_average", "butterworth", "savgol", "wiener"]:
        est = apply_classical_filter(name, noisy_val, best_params[name], clean_psd=clean_psd)
        mse = compute_mse(clean_val, est)
        snr_imp = compute_snr_improvement_db(clean_val, noisy_val, est)
        print(f"{name:<16} | {mse:>9.5f} | {snr_imp:>16.2f} | {best_params[name]}")

    with open(RESULTS_DIR / "classical_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    np.save(RESULTS_DIR / "clean_psd.npy", clean_psd)
    print("\nSaved tuned parameters -> results/classical_params.json")


if __name__ == "__main__":
    main()