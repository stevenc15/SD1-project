from scipy.io import wavfile
from scipy.fftpack import rfft, irfft, rfftfreq
import os
import sys
import numpy as np
#fourier transform for individual channe;s
def fft_filter_signal_single_channel(signal, threshold=5e3):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    result=irfft(fourier)
    return result

def fft_filter_signal(signal, threshold=5e3):
    return np.array([fft_filter_signal_single_channel(channel, threshold) for channel in signal.T]).T

def get_data_from_wav_file(filename): #return data shape [time_steps,channels]
        sample_rate, data = wavfile.read(filename)
        # print(f"Sample rate: {sample_rate}Hz|Data shape: {data.shape}")
        return data, sample_rate


