"""
Part 4: Train 1D U-Net across 5 random seeds on AWGN (0-20 dB) and benchmark
against all 4 validation-tuned classical filters across discrete test SNRs.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, compute_mse, compute_snr_improvement_db
from src.data_gen import build_dataset
from src.classical_filters import apply_classical_filter
from src.model import UNet1D


SEEDS = [0, 1, 2, 3, 4]
TEST_SNRS = [0, 5, 10, 15, 20]
EPOCHS = 25
BATCH_SIZE = 64
LR = 2e-3

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
for d in [MODELS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def train_one_seed(
    seed: int,
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    clean_val: np.ndarray,
    noisy_val: np.ndarray,
    device: torch.device,
) -> UNet1D:
    set_seed(seed)
    model = UNet1D(base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(noisy_train[:, None, :]),
        torch.from_numpy(clean_train[:, None, :]),
    )
    val_x = torch.from_numpy(noisy_val[:, None, :]).to(device)
    val_y = torch.from_numpy(clean_val[:, None, :]).to(device)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    ckpt_path = MODELS_DIR / f"unet_seed_{seed}.pt"
    torch.save(best_state, ckpt_path)
    print(f"Seed {seed} done in {elapsed:.1f}s | Best Val MSE: {best_val_loss:.5f} -> {ckpt_path}")
    return model


@torch.no_grad()
def predict_unet(model: UNet1D, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(noisy[:, None, :]).to(device)
    out = model(x).cpu().numpy()[:, 0, :]
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Part 4 on device: {device}")

    # Load validation-tuned classical parameters from Part 3
    with open(RESULTS_DIR / "classical_params.json", "r", encoding="utf-8") as f:
        classical_params = json.load(f)
    clean_psd = np.load(RESULTS_DIR / "clean_psd.npy")

    # Build train & val sets (same seeds as Part 3; strictly separate from test seeds)
    clean_train, noisy_train, _ = build_dataset(
        n_signals=2000, family="sin_chirp", snr_range_db=(0.0, 20.0), seed=1001, noise_type="awgn"
    )
    clean_val, noisy_val, _ = build_dataset(
        n_signals=400, family="sin_chirp", snr_range_db=(0.0, 20.0), seed=2001, noise_type="awgn"
    )

    n_params = sum(p.numel() for p in UNet1D(base_ch=16).parameters())
    print(f"1D U-Net parameter count: {n_params:,}\n")

    # Train 5 seeds
    models = []
    for seed in SEEDS:
        m = train_one_seed(seed, clean_train, noisy_train, clean_val, noisy_val, device)
        models.append(m)

    # Evaluate all methods across discrete test SNRs (200 unseen test signals per SNR)
    rows = []
    classical_names = ["moving_average", "butterworth", "savgol", "wiener"]

    for snr_db in TEST_SNRS:
        clean_test, noisy_test, _ = build_dataset(
            n_signals=200,
            family="sin_chirp",
            snr_range_db=(float(snr_db), float(snr_db)),
            seed=3000 + snr_db,
            noise_type="awgn",
        )

        # Classical baselines (deterministic on fixed test set -> std = 0.0)
        for name in classical_names:
            est = apply_classical_filter(name, noisy_test, classical_params[name], clean_psd=clean_psd)
            rows.append({
                "input_snr_db": snr_db,
                "method": name,
                "mse_mean": compute_mse(clean_test, est),
                "mse_std": 0.0,
                "snr_imp_mean": compute_snr_improvement_db(clean_test, noisy_test, est),
                "snr_imp_std": 0.0,
            })

        # 1D U-Net across 5 seeds
        mses, imps = [], []
        for m in models:
            est = predict_unet(m, noisy_test, device)
            mses.append(compute_mse(clean_test, est))
            imps.append(compute_snr_improvement_db(clean_test, noisy_test, est))
        rows.append({
            "input_snr_db": snr_db,
            "method": "unet_1d",
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)),
            "snr_imp_mean": float(np.mean(imps)),
            "snr_imp_std": float(np.std(imps, ddof=1)),
        })

    df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "part4_in_distribution.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Part 4: Test Performance vs. Input SNR (sin_chirp, AWGN) ---")
    for snr_db in TEST_SNRS:
        sub = df[df["input_snr_db"] == snr_db]
        print(f"\nInput SNR = {snr_db} dB:")
        for _, r in sub.iterrows():
            print(
                f"  {r['method']:<15} | MSE: {r['mse_mean']:.5f} +/- {r['mse_std']:.5f} "
                f"| SNR Imp: {r['snr_imp_mean']:+6.2f} +/- {r['snr_imp_std']:.2f} dB"
            )

    # Plot MSE and SNR improvement vs. Input SNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = {
        "moving_average": "Moving Average",
        "butterworth": "Butterworth LP",
        "savgol": "Savitzky-Golay",
        "wiener": "Wiener Filter",
        "unet_1d": "1D U-Net (5 seeds)",
    }

    for method, label in labels.items():
        sub = df[df["method"] == method]
        lw = 2.4 if method == "unet_1d" else 1.6
        axes[0].errorbar(
            sub["input_snr_db"], sub["mse_mean"], yerr=sub["mse_std"],
            marker="o", linewidth=lw, capsize=3, label=label
        )
        axes[1].errorbar(
            sub["input_snr_db"], sub["snr_imp_mean"], yerr=sub["snr_imp_std"],
            marker="o", linewidth=lw, capsize=3, label=label
        )

    axes[0].set_title("Test MSE vs. Input SNR (In-Distribution AWGN)")
    axes[0].set_xlabel("Input SNR (dB)")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("SNR Improvement vs. Input SNR (In-Distribution AWGN)")
    axes[1].set_xlabel("Input SNR (dB)")
    axes[1].set_ylabel("SNR Improvement (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig_path = FIG_DIR / "part4_snr_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nSaved Part 4 table -> {csv_path}")
    print(f"Saved Part 4 figure -> {fig_path}")


if __name__ == "__main__":
    main()