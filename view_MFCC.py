# Метод мел-частотних кепстральних коефіцієнтів (MFCC) використовується для представлення спектральних характеристик звукового сигналу у формі набора числових ознак.
# На відміну від простого поділу спектра на частотні смуги та аналізу амплітуд, MFCC моделює людське сприйняття звуку, використовуючи мел-шкалу частот,
# яка відображає нерівномірну чутливість слуху до різних частот. Після перетворення Фур’є спектр проходить через набір мел-фільтрів,
# потім обчислюються кепстральні коефіцієнти, що відображають форму спектра у стислому вигляді.
# Це дає більш стійкі, компактні та інформативні характеристики для подальшого аналізу або класифікації звуків.

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc

SR_TARGET = 16000
DURATION = 3  # секунди

files = {
    'Shahed': 'shahed24.wav',
    'FPV': 'self2_8s.wav',
    'Wind': 'audioWind4_6s.wav'
}

mfcc_features = {}

for label, path in files.items():
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != SR_TARGET:
        raise ValueError(f"{label}: неправильна частота дискретизації")
    
    y = y[:DURATION * SR_TARGET]
    # Обчислення MFCC (параметри можна коригувати)
    mfcc_feat = mfcc(signal=y, samplerate=SR_TARGET, numcep=13, nfft=2048, winlen=0.025, winstep=0.01)
    mfcc_features[label] = mfcc_feat

# Візуалізація середніх MFCC по часу
plt.figure(figsize=(12, 6))
for label, mfcc_feat in mfcc_features.items():
    mean_mfcc = np.mean(mfcc_feat, axis=0)
    plt.plot(mean_mfcc, label=label)

plt.xlabel('Номер коефіцієнта MFCC')
plt.ylabel('Середнє значення')
plt.title('Середні MFCC по часу для трьох записів')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
