import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SR_TARGET = 16000
N_FFT = 8192

files = {
    'Shahed': 'shahed22_21s.wav',
    'FPV': 'self2_8s.wav',
    'Wind': 'audioWind4_6s.wav'
}

norm_spectra = {}
band_energies = {}

# Діапазон 500-1500 Гц розбитий на 15 смуг
low_freq = 500
high_freq = 1500
num_bands = 15
band_width = (high_freq - low_freq) / num_bands
bands = [(low_freq + i*band_width, low_freq + (i+1)*band_width) for i in range(num_bands)]

for label, path in files.items():
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != SR_TARGET:
        raise ValueError(f"{label}: неправильна частота дискретизації")
    
    y = y[:6 * SR_TARGET]  # беремо 3 секунди
    y = y * np.hanning(len(y))
    yf = fft(y, n=N_FFT)
    xf = fftfreq(N_FFT, 1 / SR_TARGET)
    idx = xf >= 0
    xf = xf[idx]
    yf = np.abs(yf[idx])

    norm_yf = yf / np.max(yf)
    norm_spectra[label] = (xf, norm_yf)

    total_energy = np.sum(norm_yf[(xf >= low_freq) & (xf < high_freq)])  # енергія лише у 500-1500 Гц
    band_energy = []
    for low, high in bands:
        mask = (xf >= low) & (xf < high)
        band_energy.append(np.sum(norm_yf[mask]) / total_energy if total_energy > 0 else 0)
    band_energies[label] = band_energy

# Візуалізація розподілу енергії по 15 смугах у 500-1500 Гц
labels = list(band_energies.keys())
energies = np.array([band_energies[l] for l in labels])
band_labels = [f'{int(low)}-{int(high)}' for (low, high) in bands]

x = np.arange(len(bands))
width = 0.25

plt.figure(figsize=(14, 5))
for i, (label, e) in enumerate(zip(labels, energies)):
    plt.bar(x + i * width, e, width, label=label)

plt.xticks(x + width, band_labels, rotation=45)
plt.ylabel('Частка енергії в смузі')
plt.xlabel('Частотний діапазон, Гц')
plt.title('Розподіл енергії в діапазоні 500-1500 Гц')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
