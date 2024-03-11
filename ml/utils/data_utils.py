from scipy.io import wavfile
from scipy.fftpack import rfft, irfft, rfftfreq
import os
import sys
#fourier transform for individual channe;s
def filter_signal_single_channel(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    result=irfft(fourier)
    return result

def get_data_from_wav_file(filename): #return data shape [time_steps,channels]
        sample_rate, data = wavfile.read(filename)
        # print(f"Sample rate: {sample_rate}Hz|Data shape: {data.shape}")
        return data, sample_rate

def to_tensor(data): #convert to tensor
    pass
