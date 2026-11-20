"""
Part 6: Uncertainty Estimation & Calibration via 5-Member Deep Ensemble.

Uses the 5 independently trained 1D U-Net checkpoints from Part 4:
  - Predictive mean:        mu(t)    = mean_m(x_hat_m(t))
  - Epistemic uncertainty:  sigma(t) = std_m(x_hat_m(t))  (ddof=1)

Evaluates:
  1. Correlation (Spearman & Pearson) between predicted uncertainty sigma and actual |error|
     both pointwise (across all samples) and signal-level (RMSE vs. mean sigma).
  2. Empirical interval coverage at +/-1 sigma (ideal 68.3%) and +/-2 sigma (ideal 95.4%),
     plus normalized error variance E[(e/sigma)^2] (ideal = 1.0) to check calibration honestly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import torch

from src.data_gen import build_dataset, FS
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
def ensemble_predict(
    models: list[UNet1D], noisy: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ensemble_mean, ensemble_std) with shape (M, N)."""
    x = torch.from_numpy(noisy[:, None, :]).to(device)
    preds = np.stack([m(x).cpu().numpy()[:, 0, :] for m in models], axis=0)  # (K, M, N)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0, ddof=1)
    return mean, std


def evaluate_uncertainty(clean: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    abs_err = np.abs(clean - mean)
    flat_err = abs_err.ravel()
    flat_std = std.ravel()

    # Pointwise correlation
    sp_pt, _ = spearmanr(flat_std, flat_err)
    pe_pt, _ = pearsonr(flat_std, flat_err)

    # Signal-level correlation (per-signal RMSE vs per-signal RMS uncertainty)
    sig_rmse = np.sqrt(np.mean((clean - mean) ** 2, axis=-1))
    sig_unc = np.sqrt(np.mean(std ** 2, axis=-1))
    sp_sig, _ = spearmanr(sig_unc, sig_rmse)
    pe_sig, _ = pearsonr(sig_unc, sig_rmse)

    # Empirical coverage of +/-1 sigma and +/-2 sigma intervals
    cov_1 = float(np.mean(abs_err <= 1.0 * std))
    cov_2 = float(np.mean(abs_err <= 2.0 * std))

    # Ratio of actual RMSE to mean predicted std (1.0 = well-calibrated, >1.0 = under-dispersed/overconfident)
    z_rms = float(np.sqrt(np.mean((clean - mean) ** 2)) / (np.sqrt(np.mean(std ** 2)) + 1e-12))

    return {
        "ensemble_mse": compute_mse(clean, mean),
        "mean_sigma": float(np.mean(flat_std)),
        "mean_abs_err": float(np.mean(flat_err)),
        "spearman_point": float(sp_pt),
        "pearson_point": float(pe_pt),
        "spearman_signal": float(sp_sig),
        "pearson_signal": float(pe_sig),
        "cov_1sigma": cov_1,
        "cov_2sigma": cov_2,
        "rmse_to_sigma_ratio": z_rms,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(device)

    datasets = [
        ("ID_AWGN_0_20dB",    "sin_chirp", "awgn",      (0.0, 20.0),  6001),
        ("OOD_LowSNR_-5dB",   "sin_chirp", "awgn",      (-5.0, -5.0), 6002),
        ("OOD_Impulse_10dB",  "sin_chirp", "impulsive", (10.0, 10.0), 6003),
        ("OOD_BPSK_10dB",     "bpsk",      "awgn",      (10.0, 10.0), 6004),
    ]

    rows = []
    cached_id = None

    print("\n--- Part 6: Deep Ensemble (K=5) Uncertainty & Calibration ---")
    for label, family, noise_type, snr_range, seed in datasets:
        clean, noisy, _ = build_dataset(
            n_signals=250, family=family, snr_range_db=snr_range, seed=seed, noise_type=noise_type
        )
        mean, std = ensemble_predict(models, noisy, device)
        metrics = evaluate_uncertainty(clean, mean, std)
        metrics["snr_imp_db"] = compute_snr_improvement_db(clean, noisy, mean)
        metrics["scenario"] = label
        rows.append(metrics)

        if label == "ID_AWGN_0_20dB":
            cached_id = (clean, noisy, mean, std)

        print(
            f"\n{label}:\n"
            f"  Ensemble MSE: {metrics['ensemble_mse']:.5f} | SNR Imp: {metrics['snr_imp_db']:+.2f} dB\n"
            f"  Mean |Err|:   {metrics['mean_abs_err']:.4f} | Mean Sigma: {metrics['mean_sigma']:.4f} "
            f"(RMSE/Sigma = {metrics['rmse_to_sigma_ratio']:.2f}x)\n"
            f"  Pointwise r:  Spearman={metrics['spearman_point']:.3f}, Pearson={metrics['pearson_point']:.3f}\n"
            f"  Signal-lvl r: Spearman={metrics['spearman_signal']:.3f}, Pearson={metrics['pearson_signal']:.3f}\n"
            f"  Coverage:     +/-1s={metrics['cov_1sigma']*100:.1f}% (ideal 68.3%), "
            f"+/-2s={metrics['cov_2sigma']*100:.1f}% (ideal 95.4%)"
        )

    df = pd.DataFrame(rows)
    cols = [
        "scenario", "ensemble_mse", "snr_imp_db", "mean_abs_err", "mean_sigma",
        "rmse_to_sigma_ratio", "spearman_point", "pearson_point",
        "spearman_signal", "pearson_signal", "cov_1sigma", "cov_2sigma"
    ]
    df = df[cols]
    csv_path = TABLE_DIR / "part6_uncertainty_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Plot 3-panel uncertainty & calibration figure on ID_AWGN_0_20dB
    clean, noisy, mean, std = cached_id
    flat_err = np.abs(clean - mean).ravel()
    flat_std = std.ravel()

    # Quantile-binned Error vs Uncertainty
    n_bins = 15
    edges = np.quantile(flat_std, np.linspace(0, 1, n_bins + 1))
    bin_sigma, bin_rmse = [], []
    for i in range(n_bins):
        mask = (flat_std >= edges[i]) & (flat_std <= edges[i + 1])
        if np.any(mask):
            bin_sigma.append(np.sqrt(np.mean(flat_std[mask] ** 2)))
            bin_rmse.append(np.sqrt(np.mean(flat_err[mask] ** 2)))

    # Calibration curve across z-multipliers
    ks = np.linspace(0.25, 4.0, 20)
    from scipy.stats import norm
    ideal_cov = 2 * norm.cdf(ks) - 1
    emp_cov = [np.mean(flat_err <= k * flat_std) for k in ks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Binned RMSE vs Predicted Uncertainty
    axes[0].plot(bin_sigma, bin_rmse, "o-", linewidth=2, label="Observed bin RMSE")
    max_val = max(max(bin_sigma), max(bin_rmse))
    axes[0].plot([0, max_val], [0, max_val], "k--", label="Ideal calibration (y = x)")
    axes[0].set_title("Error vs. Predicted Uncertainty (ID)")
    axes[0].set_xlabel("Predicted Ensemble Std (RMS)")
    axes[0].set_ylabel("Actual Error (RMSE)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel 2: Calibration curve
    axes[1].plot(ideal_cov * 100, np.array(emp_cov) * 100, "s-", linewidth=2, label="Raw Ensemble")
    axes[1].plot([0, 100], [0, 100], "k--", label="Perfect Calibration")
    axes[1].set_title("Interval Coverage Calibration (ID)")
    axes[1].set_xlabel("Expected Coverage (%)")
    axes[1].set_ylabel("Observed Coverage (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    # Panel 3: Example signal with +/-2 sigma band
    idx = 0
    t = np.arange(200) / FS
    axes[2].plot(t, clean[idx, :200], "k", linewidth=1.3, label="Clean")
    axes[2].plot(t, mean[idx, :200], "C0", linewidth=1.2, label="Ensemble Mean")
    axes[2].fill_between(
        t,
        mean[idx, :200] - 2 * std[idx, :200],
        mean[idx, :200] + 2 * std[idx, :200],
        color="C0",
        alpha=0.25,
        label="+/- 2 sigma (epistemic)",
    )
    axes[2].set_title("Sample Denoised Waveform + Uncertainty")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig_path = FIG_DIR / "part6_uncertainty_calibration.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nSaved Part 6 table  -> {csv_path}")
    print(f"Saved Part 6 figure -> {fig_path}")


if __name__ == "__main__":
    main()