import soundfile as sf
import numpy as np
from scipy.signal import butter, sosfilt

def bandpass_filter(data, fs, lowcut=500, highcut=6000, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, data)

def parabolic_interpolation(corr, index):
    if index <= 0 or index >= len(corr) - 1:
        return 0.0
    alpha = corr[index - 1]
    beta = corr[index]
    gamma = corr[index + 1]
    numerator = alpha - gamma
    denominator = 2 * (alpha - 2 * beta + gamma)
    if denominator == 0:
        return 0.0
    return 0.5 * numerator / denominator

def estimate_delay(signal1, signal2, fs, max_delay_s=0.002):
    max_delay = int(max_delay_s * fs)
    corr = np.correlate(signal1, signal2, mode='full')
    mid = len(corr) // 2
    corr_limited = corr[mid - max_delay : mid + max_delay + 1]
    peak_index = np.argmax(corr_limited)
    interp_offset = parabolic_interpolation(corr_limited, peak_index)
    delay_index = peak_index - max_delay + interp_offset
    delay_time = delay_index / fs
    return delay_time

def estimate_angle(delay_time, mic_distance, sound_speed=343):
    max_possible = mic_distance / sound_speed
    if abs(delay_time) > max_possible:
        print("Попередження: затримка перевищує фізично допустиму межу.")
    val = delay_time * sound_speed / mic_distance
    val = np.clip(val, -1, 1)
    angle_rad = np.arccos(val)
    return np.degrees(angle_rad)

def main(stereo_file, mic_distance=0.2, sample_rate=44100):
    data, fs = sf.read(stereo_file)

    if fs != sample_rate:
        print(f"Увага: очікувана частота дискретизації {sample_rate} Гц, отримано {fs} Гц")

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Очікується стереофайл із двома каналами")

    left = data[:, 0]
    right = data[:, 1]

    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    left_filt = bandpass_filter(left, fs)
    right_filt = bandpass_filter(right, fs)

    delay = estimate_delay(left_filt, right_filt, fs)
    angle = estimate_angle(delay, mic_distance)

    print(f"Затримка сигналу між мікрофонами: {delay*1000:.3f} мс")
    print(f"Орієнтовний кут джерела звуку відносно лінії мікрофонів: {angle:.2f}°")

if __name__ == "__main__":
    main("stereo_20250612_171210.wav", mic_distance=0.023, sample_rate=48000)
    main("stereo_20250612_171217.wav", mic_distance=0.023, sample_rate=48000)
    main("stereo_20250612_171224.wav", mic_distance=0.023, sample_rate=48000)
