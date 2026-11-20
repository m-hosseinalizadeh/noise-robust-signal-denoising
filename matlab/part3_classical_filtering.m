% Part 3 (MATLAB Companion): Classical Filtering on Noisy Multi-Tone Signal
% Compatible with MATLAB R2018b (handles missing Signal Processing Toolbox gracefully)
clear; clc; close all;
rng(42); % Fixed seed for reproducibility

if ~exist('../results/figures', 'dir')
    mkdir('../results/figures');
end

% Generate unit-RMS multi-tone signal (50, 120, 180 Hz) at fs = 1000 Hz
fs = 1000;
N = 512;
t = (0:N-1) / fs;
clean = 0.5*sin(2*pi*50*t) + 0.3*sin(2*pi*120*t) + 0.2*sin(2*pi*180*t);
clean = clean / sqrt(mean(clean.^2));

% Add AWGN at exact 5 dB SNR
target_snr_db = 5.0;
noise = randn(1, N);
noise = noise * sqrt(mean(clean.^2) / (10^(target_snr_db/10) * mean(noise.^2)));
noisy = clean + noise;

snr_fn = @(ref, est) 10 * log10(mean(ref.^2) / mean((ref - est).^2));
snr_in = snr_fn(clean, noisy);

% 1. Zero-phase Moving Average (window = 3) using conv
h_ma = ones(1, 3) / 3;
est_ma = conv(noisy, h_ma, 'same');
snr_ma = snr_fn(clean, est_ma);

% 2. Zero-phase Butterworth LP (cutoff = 200 Hz, order = 6) if toolbox available
if exist('butter', 'file') && exist('filtfilt', 'file')
    [b, a] = butter(6, 200 / (fs/2), 'low');
    est_lp = filtfilt(b, a, noisy);
    lp_label = 'Butterworth LP (200 Hz, ord 6)';
else
    % Fallback: zero-phase frequency-domain low-pass at 200 Hz
    spec = fft(noisy);
    freqs = (0:N-1) * (fs / N);
    mask = (freqs <= 200) | (freqs >= fs - 200);
    est_lp = real(ifft(spec .* mask));
    lp_label = 'FFT Ideal LP (200 Hz)';
end
snr_lp = snr_fn(clean, est_lp);

fprintf('\n--- MATLAB Classical Filtering Demo (5 dB AWGN) ---\n');
fprintf('Input SNR:       %6.2f dB\n', snr_in);
fprintf('Moving Avg (3):  %6.2f dB (Imp: %+5.2f dB)\n', snr_ma, snr_ma - snr_in);
fprintf('%-15s: %6.2f dB (Imp: %+5.2f dB)\n', lp_label, snr_lp, snr_lp - snr_in);

figure('Name', 'MATLAB Classical Filtering');
plot(t, noisy, 'Color', [0.8 0.8 0.8], 'LineWidth', 1.0); hold on;
plot(t, clean, 'k', 'LineWidth', 1.4);
plot(t, est_ma, 'r--', 'LineWidth', 1.1);
plot(t, est_lp, 'b', 'LineWidth', 1.2);
grid on; xlim([0 0.15]);
xlabel('Time (s)'); ylabel('Amplitude');
title('MATLAB Classical Filtering (5 dB AWGN, First 150 ms)');
legend('Noisy (5 dB)', 'Clean', 'Moving Avg (3)', lp_label, 'Location', 'NorthEast');
saveas(gcf, '../results/figures/part3_filtering_matlab.png');