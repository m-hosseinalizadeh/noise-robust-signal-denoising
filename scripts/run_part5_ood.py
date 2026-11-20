"""
Part 5: Out-of-Distribution (OOD) Robustness and Generalization Stress Tests.

Evaluates all 4 validation-tuned classical filters and the 5-seed 1D U-Net on
conditions never seen during training:
  1. Colored noise (1/f^1.5) at 10 dB
  2. Impulsive noise (Bernoulli-Gaussian) at 10 dB
  3. Low SNR outside training range (-5 dB AWGN)
  4. High SNR outside training range (30 dB AWGN)
  5. Unseen signal family (pulse-shaped BPSK baseband) at 10 dB AWGN
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_gen import build_dataset, FS
from src.classical_filters import apply_classical_filter
from src.model import UNet1D
from src.utils import compute_mse, compute_snr_improvement_db


SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


def load_ensemble(device: torch.device) -> list[UNet1D]:
    models = []
    for seed in SEEDS:
        m = UNet1D(base_ch=16).to(device)
        m.load_state_dict(torch.load(MODELS_DIR / f"unet_seed_{seed}.pt", map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_unet(model: UNet1D, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(noisy[:, None, :]).to(device)
    return model(x).cpu().numpy()[:, 0, :]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(RESULTS_DIR / "classical_params.json", "r", encoding="utf-8") as f:
        classical_params = json.load(f)
    clean_psd = np.load(RESULTS_DIR / "clean_psd.npy")
    models = load_ensemble(device)

    # Define OOD test suites (+ In-Dist 10 dB reference)
    scenarios = [
        ("ID_AWGN_10dB",      "sin_chirp", "awgn",      10.0, 5001),
        ("OOD_Colored_10dB",  "sin_chirp", "colored",   10.0, 5002),
        ("OOD_Impulse_10dB",  "sin_chirp", "impulsive", 10.0, 5003),
        ("OOD_LowSNR_-5dB",   "sin_chirp", "awgn",      -5.0, 5004),
        ("OOD_HighSNR_30dB",  "sin_chirp", "awgn",      30.0, 5005),
        ("OOD_BPSK_10dB",     "bpsk",      "awgn",      10.0, 5006),
    ]

    classical_names = ["moving_average", "butterworth", "savgol", "wiener"]
    rows = []
    waveform_examples = {}

    for label, family, noise_type, snr_db, seed in scenarios:
        clean_test, noisy_test, _ = build_dataset(
            n_signals=200,
            family=family,
            snr_range_db=(snr_db, snr_db),
            seed=seed,
            noise_type=noise_type,
        )

        # Classical baselines
        est_map = {}
        for name in classical_names:
            est = apply_classical_filter(name, noisy_test, classical_params[name], clean_psd=clean_psd)
            est_map[name] = est
            rows.append({
                "scenario": label,
                "family": family,
                "noise_type": noise_type,
                "input_snr_db": snr_db,
                "method": name,
                "mse_mean": compute_mse(clean_test, est),
                "mse_std": 0.0,
                "snr_imp_mean": compute_snr_improvement_db(clean_test, noisy_test, est),
                "snr_imp_std": 0.0,
            })

        # 1D U-Net (mean +/- std across 5 seeds)
        mses, imps, preds = [], [], []
        for m in models:
            est = predict_unet(m, noisy_test, device)
            preds.append(est)
            mses.append(compute_mse(clean_test, est))
            imps.append(compute_snr_improvement_db(clean_test, noisy_test, est))

        rows.append({
            "scenario": label,
            "family": family,
            "noise_type": noise_type,
            "input_snr_db": snr_db,
            "method": "unet_1d",
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)),
            "snr_imp_mean": float(np.mean(imps)),
            "snr_imp_std": float(np.std(imps, ddof=1)),
        })

        # Save one example for qualitative plotting
        if label in {"OOD_Colored_10dB", "OOD_Impulse_10dB", "OOD_BPSK_10dB"}:
            waveform_examples[label] = (
                clean_test[0], noisy_test[0], est_map["wiener"][0], preds[0][0]
            )

    df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "part5_ood_robustness.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Part 5: OOD Robustness & Generalization (SNR Improvement in dB) ---")
    for label, _, _, _, _ in scenarios:
        sub = df[df["scenario"] == label]
        print(f"\nScenario: {label}")
        for _, r in sub.iterrows():
            print(
                f"  {r['method']:<15} | MSE: {r['mse_mean']:.5f} +/- {r['mse_std']:.5f} "
                f"| SNR Imp: {r['snr_imp_mean']:+6.2f} +/- {r['snr_imp_std']:.2f} dB"
            )

    # Bar chart of SNR improvement across scenarios
    methods = ["moving_average", "butterworth", "savgol", "wiener", "unet_1d"]
    pretty = ["Moving Avg", "Butterworth", "Savitzky-Golay", "Wiener", "1D U-Net"]
    scen_labels = [s[0] for s in scenarios]

    x = np.arange(len(scen_labels))
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (m, p) in enumerate(zip(methods, pretty)):
        sub = df[df["method"] == m].set_index("scenario").loc[scen_labels]
        ax.bar(
            x + (i - 2) * width,
            sub["snr_imp_mean"],
            width,
            yerr=sub["snr_imp_std"],
            capsize=2,
            label=p,
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_labels, rotation=15, ha="right")
    ax.set_ylabel("SNR Improvement (dB)")
    ax.set_title("In-Distribution vs. Out-of-Distribution SNR Improvement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=5, loc="upper right", fontsize=9)
    plt.tight_layout()
    bar_path = FIG_DIR / "part5_ood_comparison.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()

    # Qualitative OOD waveforms
    t = np.arange(512) / FS
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, label in zip(axes, ["OOD_Colored_10dB", "OOD_Impulse_10dB", "OOD_BPSK_10dB"]):
        c, y, w_est, u_est = waveform_examples[label]
        ax.plot(t, y, color="lightgray", linewidth=1.0, label="Noisy")
        ax.plot(t, c, color="black", linewidth=1.3, label="Clean")
        ax.plot(t, w_est, color="tab:orange", linestyle="--", linewidth=1.1, label="Wiener")
        ax.plot(t, u_est, color="tab:blue", linewidth=1.1, alpha=0.85, label="1D U-Net")
        ax.set_title(label)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    wf_path = FIG_DIR / "part5_ood_waveforms.png"
    plt.savefig(wf_path, dpi=150)
    plt.close()

    print(f"\nSaved Part 5 table   -> {csv_path}")
    print(f"Saved Part 5 figures -> {bar_path}, {wf_path}")


if __name__ == "__main__":
    main()