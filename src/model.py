"""
Part 4: Lightweight 3-Level Residual 1D U-Net for Signal Denoising.

Predicts a residual correction (x_hat = y + net(y)), which stabilizes training
and preserves signal scale across wide input SNR ranges.
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, base_ch: int = 16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Encoder
        self.enc1 = ConvBlock1D(1, c1, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(c1, c2, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(c2, c3, kernel_size=5)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck (length 512 -> 64, large receptive field for low-freq tones)
        self.bottleneck = ConvBlock1D(c3, c3, kernel_size=5)

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose1d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(c3 * 2, c2, kernel_size=5)

        self.up2 = nn.ConvTranspose1d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(c2 * 2, c1, kernel_size=5)

        self.up1 = nn.ConvTranspose1d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(c1 * 2, c1, kernel_size=5)

        self.head = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(y)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Residual denoising: estimate correction to noisy input
        return y + self.head(d1)