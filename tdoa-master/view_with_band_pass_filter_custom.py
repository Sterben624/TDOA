import sys
import math
import numpy as np
import wave
from scipy.signal import windows
from gcc_phat import gcc_phat  # Залишено імпорт, якщо функція винесена окремо

RATE = 44100
FRAMES = int(RATE / 4)

window = windows.hann(FRAMES)

sound_speed = 343.2
distance = 0.22

max_tau = distance / sound_speed
direction_n = int(max_tau * RATE)


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    n = sig.shape[0] + refsig.shape[0]

    low_cutoff_bin = int((250 * n) / RATE)
    high_cutoff_bin = int((4000 * n) / RATE)

    SIG = np.fft.rfft(sig, n=n)
    SIG[:low_cutoff_bin + 1] = 0
    SIG[-high_cutoff_bin:] = 0

    REFSIG = np.fft.rfft(refsig, n=n)
    REFSIG[:low_cutoff_bin + 1] = 0
    REFSIG[-high_cutoff_bin:] = 0

    T = SIG[low_cutoff_bin + 1:-high_cutoff_bin] * np.conj(REFSIG[low_cutoff_bin + 1:-high_cutoff_bin])
    T /= np.abs(T)
    SIG[low_cutoff_bin + 1:-high_cutoff_bin] = T

    cc = np.fft.irfft(SIG, n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    return tau, cc


def process_wav_file(filepath):
    with wave.open(filepath, 'rb') as wf:
        if wf.getnchannels() != 2:
            raise ValueError('Очікується 2-канальний WAV файл')
        if wf.getframerate() != RATE:
            raise ValueError(f'Очікувана частота дискретизації: {RATE} Гц')

        total_frames = wf.getnframes()
        audio_data = wf.readframes(total_frames)
        signal = np.frombuffer(audio_data, dtype=np.int16)
        ch1 = signal[0::2]
        ch2 = signal[1::2]

        min_len = min(len(ch1), len(ch2), len(window))
        ch1_windowed = ch1[:min_len] * window[:min_len]
        ch2_windowed = ch2[:min_len] * window[:min_len]

        tau, cc = gcc_phat(ch1_windowed, ch2_windowed, fs=RATE, max_tau=max_tau, interp=1)
        theta = math.asin(tau / max_tau) * 180 / math.pi
        print('Визначений кут приходу (theta): {:.2f} градусів'.format(theta))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Використання: python script.py <шлях_до_wav_файлу>')
        sys.exit(1)

    wav_path = sys.argv[1]
    process_wav_file(wav_path)
