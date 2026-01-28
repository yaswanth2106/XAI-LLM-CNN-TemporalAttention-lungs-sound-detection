import numpy as np
import librosa
import io
MAX_LEN = 512
N_MFCC = 40

def audio_to_mfcc(file):
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad = np.zeros((N_MFCC, MAX_LEN))
        start = (MAX_LEN - mfcc.shape[1]) // 2
        pad[:, start:start + mfcc.shape[1]] = mfcc
        mfcc = pad
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / \
           (mfcc.std(axis=1, keepdims=True) + 1e-6)

    return mfcc
