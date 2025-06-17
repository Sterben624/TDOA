import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Налаштування
SR_TARGET = 16000
N_FFT = 8192

files = {
    'Shahed': 'shahed1.wav',
    'FPV': 'self1.wav',
    'Wind': 'audioWind1.wav'
}

plt.figure(figsize=(12, 5))

for label, path in files.items():
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != SR_TARGET:
        raise ValueError(f"Частота дискретизації {path} повинна бути {SR_TARGET} Гц")
    
    # Відбір фрагменту (наприклад, 2 секунди максимум)
    y = y[:2 * SR_TARGET]

    # Вікно Хеннінга для зменшення побічних лобів
    window = np.hanning(len(y))
    y_win = y * window

    # FFT
    yf = fft(y_win, n=N_FFT)
    xf = fftfreq(N_FFT, 1 / SR_TARGET)

    # Позитивні частоти
    idx = xf >= 0
    xf = xf[idx]
    yf = np.abs(yf[idx])

    # Перехід в логарифмічну шкалу
    amplitude_dB = 20 * np.log10(yf + 1e-12)
    plt.plot(xf, amplitude_dB, label=label)

plt.xlim([0, 4000])  # типові сигнали в межах до 4 кГц
plt.xlabel('Частота, Гц')
plt.ylabel('Амплітуда, дБ')
plt.title('Амплітудний спектр сигналів')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
