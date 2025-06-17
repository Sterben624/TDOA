import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SR_TARGET = 16000
N_FFT = 8192

files = {
    'Shahed': 'shahed24.wav',
    'FPV': 'self2_8s.wav',
    'Wind': 'audioWind4_6s.wav'
}

norm_spectra = {}
band_energies = {}

# Розширені частотні діапазони (до 8000 Гц)
bands = [
    (0, 500), (500, 1500), (1500, 3000),
    (3000, 4000), (4000, 5500), (5500, 7000), (7000, 8000)
]

for label, path in files.items():
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != SR_TARGET:
        raise ValueError(f"{label}: неправильна частота дискретизації")
    
    y = y[:3 * SR_TARGET]
    y = y * np.hanning(len(y))
    yf = fft(y, n=N_FFT)
    xf = fftfreq(N_FFT, 1 / SR_TARGET)
    idx = xf >= 0
    xf = xf[idx]
    yf = np.abs(yf[idx])

    norm_yf = yf / np.max(yf)
    norm_spectra[label] = (xf, norm_yf)

    total_energy = np.sum(norm_yf)
    band_energy = []
    for low, high in bands:
        mask = (xf >= low) & (xf < high)
        band_energy.append(np.sum(norm_yf[mask]) / total_energy)
    band_energies[label] = band_energy

# Графік нормалізованого спектра до 8000 Гц
plt.figure(figsize=(12, 5))
for label, (xf, norm_yf) in norm_spectra.items():
    plt.plot(xf, norm_yf, label=label)
plt.xlim([0, 8000])
plt.xlabel('Частота, Гц')
plt.ylabel('Нормалізована амплітуда')
plt.title('Нормалізований амплітудний спектр')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Гістограма розподілу енергії по діапазонах до 8000 Гц
labels = list(band_energies.keys())
energies = np.array([band_energies[l] for l in labels])
band_labels = [f'{low}-{high} Гц' for (low, high) in bands]

x = np.arange(len(bands))
width = 0.25

plt.figure(figsize=(12, 4))
for i, (label, e) in enumerate(zip(labels, energies)):
    plt.bar(x + i * width, e, width, label=label)

plt.xticks(x + width, band_labels, rotation=45)
plt.ylabel('Частка енергії')
plt.title('Розподіл енергії по частотних діапазонах')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
