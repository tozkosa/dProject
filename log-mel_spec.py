import numpy as np
import librosa
import librosa.display
import os
from matplotlib import pyplot as plt
import soundfile as sf

#audio_path = 'D:/daon_data/nagoya_20210727_cutout_5ms/crack_1/train/small/normal'
audio_path = '/home/tozeki/daon/nagoya2021/nagoya_20210727_cutout_5ms/crack_1/train/small/defect'
duration = 0.005

def load_wav_file(path):
    print(wav_path)
    data, sr = librosa.load(wav_path, sr=None)
    print(f'data shape = {data.shape}')
    print(f'original sr = {sr}')

    return data, sr

def plot_wave_fft(data, sr, freq, amp_normal):
    plt.figure(figsize=(8, 8))

    # wave graph
    x_axis = np.arange(0, data.shape[0])/sr
    plt.subplot(2,1,1)
    plt.plot(x_axis, data)
    plt.title("raw waveform")
    plt.grid()

    # Fourier graph
    N = len(data)
    plt.subplot(2,1,2)
    plt.plot(freq[:int(N/2)+1], amp_normal[:int(N/2)+1])
    plt.grid()
    plt.title('Fast Fourier Transform')

def fourier_trans(data, sr):
    F = np.fft.fft(data)
    amp = np.abs(F)
    freq = np.linspace(0, sr, data.shape[0])

    # with normalization
    N = len(data)
    amp_normal = amp / (N / 2)
    amp_normal[0] /= 2
    
    return amp_normal, freq

def librosa_stft(data):
    D = librosa.stft(data, 
                    n_fft=256, 
                    win_length=256,
                    hop_length=8)
    print(type(D), D.shape)
    S, phase = librosa.magphase(D)
    #S = librosa.amplitude_to_db(S)
    plt.figure()
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz')
    #plt.ylim(0, 20000)
    plt.colorbar()

def matplotlib_stft(wav_path):
    plt.figure()
    #plt.subplot(2,2,2)
    data, sr = sf.read(wav_path)
    n_fft = 256
    hop_len = 8

    spectrum, freqs, t, im = plt.specgram(data, 
                                        NFFT=n_fft,
                                        Fs=sr,
                                        noverlap=n_fft-hop_len,
                                        scale='linear',
                                        mode='magnitude')
    plt.colorbar()
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency[Hz]')
    plt.tight_layout()
    print('spectrum shape: ', spectrum.shape)
    print('spectrum type: ', type(spectrum))

if __name__ == "__main__":

    # Load wav file
    files = os.listdir(audio_path)
    wav_path = os.path.join(audio_path, files[1])
    data, sr = load_wav_file(wav_path)
    N = len(data)
    
    
    # Fourier transform with normalization
    amp_normal, freq = fourier_trans(data, sr)
    
    # graph
    plot_wave_fft(data, sr, freq, amp_normal)

    # Short-Time Fourier Transform
    librosa_stft(data)
    matplotlib_stft(wav_path)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(y=data,
                                        sr=sr,
                                        n_mels=64,
                                        n_fft=256,
                                        win_length=256,
                                        hop_length=8)
    print(mel.shape)
    plt.figure()
    librosa.display.specshow(mel,
                            x_axis='time',
                            y_axis='linear',
                            sr=sr,
                            hop_length=8)
    plt.colorbar(format='%+2.0f')
    plt.title('mel spectrogram')
    plt.tight_layout()

    # log-mel spectrogram
    mel = librosa.feature.melspectrogram(y=data,
                                        sr=sr,
                                        n_mels=64,
                                        n_fft=256,
                                        win_length=256,
                                        hop_length=8)
    log_mel = np.log(mel)
    print(mel.shape)
    plt.figure()
    librosa.display.specshow(log_mel,
                            x_axis='time',
                            y_axis='linear',
                            sr=sr,
                            hop_length=8)
    plt.colorbar(format='%+2.0f')
    plt.title('log-mel spectrogram')
    plt.tight_layout()
    
    plt.show()
