% Part 1: Sampling, Aliasing, and Uniform Quantization (MATLAB / Octave)
clear; clc; close all;

if ~exist('../results/figures', 'dir')
    mkdir('../results/figures');
end

%% 1. Aliasing Demonstration
% Multi-tone signal: 50 Hz, 120 Hz, 200 Hz (Nyquist rate = 400 Hz)
duration = 1.0;

fs_high = 1000;
t_high = 0 : 1/fs_high : duration - 1/fs_high;
x_high = 0.5*sin(2*pi*50*t_high) + 0.3*sin(2*pi*120*t_high) + 0.2*sin(2*pi*200*t_high);

fs_low = 300; % Below Nyquist -> 200 Hz aliases to |300 - 200| = 100 Hz
t_low = 0 : 1/fs_low : duration - 1/fs_low;
x_low = 0.5*sin(2*pi*50*t_low) + 0.3*sin(2*pi*120*t_low) + 0.2*sin(2*pi*200*t_low);

% Single-sided FFT spectra
N_h = length(x_high);
f_h = (0:floor(N_h/2)) * (fs_high / N_h);
mag_h = abs(fft(x_high)) * (2 / N_h);
mag_h = mag_h(1:length(f_h)); mag_h(1) = mag_h(1)/2;

N_l = length(x_low);
f_l = (0:floor(N_l/2)) * (fs_low / N_l);
mag_l = abs(fft(x_low)) * (2 / N_l);
mag_l = mag_l(1:length(f_l)); mag_l(1) = mag_l(1)/2;

figure('Name', 'Part 1: Aliasing');
subplot(2,1,1);
stem(f_h, mag_h, 'Marker', 'none'); xlim([0 300]); grid on;
title('Above Nyquist (fs = 1000 Hz): True Peaks at 50, 120, 200 Hz');
xlabel('Frequency (Hz)'); ylabel('Magnitude');

subplot(2,1,2);
stem(f_l, mag_l, 'r', 'Marker', 'none'); xlim([0 160]); grid on;
title('Below Nyquist (fs = 300 Hz): 200 Hz Aliases to 100 Hz');
xlabel('Frequency (Hz)'); ylabel('Magnitude');

saveas(gcf, '../results/figures/part1_aliasing_matlab.png');

%% 2. Quantization and SQNR vs. Bit Depth
fs_q = 4096;
t_q = 0 : 1/fs_q : 1.0 - 1/fs_q;
x_sine = sin(2*pi*53*t_q);
x_multi = 0.5*sin(2*pi*50*t_q) + 0.3*sin(2*pi*120*t_q) + 0.2*sin(2*pi*200*t_q);
x_multi = x_multi / max(abs(x_multi));

bits = [4, 6, 8, 12];
sqnr_theory = 6.02 * bits + 1.76;
sqnr_sine = zeros(size(bits));
sqnr_multi = zeros(size(bits));

fprintf('\nBits | Theory (dB) | Single Sine (dB) | Multi-tone (dB)\n');
fprintf('--------------------------------------------------------\n');
for i = 1:length(bits)
    N = bits(i);
    step = 2.0 / (2^N);

    % Quantize single sine
    xs_clip = min(max(x_sine, -1.0), 1.0 - step);
    qs = round(xs_clip / step) * step;
    sqnr_sine(i) = 10 * log10(mean(x_sine.^2) / mean((x_sine - qs).^2));

    % Quantize multi-tone
    xm_clip = min(max(x_multi, -1.0), 1.0 - step);
    qm = round(xm_clip / step) * step;
    sqnr_multi(i) = 10 * log10(mean(x_multi.^2) / mean((x_multi - qm).^2));

    fprintf('%4d | %11.2f | %16.2f | %15.2f\n', N, sqnr_theory(i), sqnr_sine(i), sqnr_multi(i));
end

figure('Name', 'Part 1: Quantization SQNR');
plot(bits, sqnr_theory, 'k--', 'LineWidth', 1.8); hold on;
plot(bits, sqnr_sine, 'bo-', 'LineWidth', 1.5);
plot(bits, sqnr_multi, 'rs-', 'LineWidth', 1.5);
grid on; xlabel('Bit Depth (N)'); ylabel('SQNR (dB)');
title('Measured vs. Theoretical SQNR (MATLAB)');
legend('Theory: 6.02N + 1.76 dB', 'Single Sinusoid', 'Multi-tone', 'Location', 'NorthWest');
saveas(gcf, '../results/figures/part1_quantization_sqnr_matlab.png');