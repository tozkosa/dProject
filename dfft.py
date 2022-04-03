import numpy as np
import librosa
import librosa.display
import os
from matplotlib import pyplot as plt

audio_path = 'D:/daon_data/nagoya_20210727_cutout_5ms/crack_1/train/small/normal'
duration = 0.005

if __name__ == "__main__":

    # Load wav file
    files = os.listdir(audio_path)
    wav_file = os.path.join(audio_path, files[10])
    print(wav_file)

    data, sr = librosa.load(wav_file, sr=96000)
    x_axis = np.arange(0, data.shape[0])/sr
    
    print(f'data shape = {data.shape[0]}')


    plt.figure(figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.plot(x_axis, data)
    plt.title("raw waveform")
    plt.grid()

    # Fourier transform
    F = np.fft.fft(data)
    amp = np.abs(F)
    freq = np.linspace(0, sr, data.shape[0])

    # with normalization
    N = len(data)
    amp_normal = amp / (N / 2)
    amp_normal[0] /= 2
    
    plt.subplot(2,1,2)
    plt.plot(freq[:int(data.shape[0]/2)+1], amp_normal[:int(data.shape[0]/2)+1])
    plt.grid()
    plt.title('Fast Fourier Transform')

    # Short-Time Fourier Transform
    S_F = librosa.stft(data, 
                       n_fft=280,
                       win_length=64,
                       hop_length=16)
    bins = librosa.fft_frequencies(sr=96000, n_fft=512)
    print('bins: ', bins.shape)
    
    amp = np.abs(S_F)
    amp_normal = amp / (N / 2)
    amp_normal[0] /= 2

    amp_normal_db = librosa.amplitude_to_db(amp_normal, ref=0)
    amp_normal_db_max = librosa.amplitude_to_db(amp_normal, ref=np.max)
    amp_normal_db_max2 = librosa.amplitude_to_db(amp, ref=np.max)

    plt.figure()
    plt.subplot(2,2,1)
    librosa.display.specshow(amp_normal,
                             sr=96000,
                             hop_length=96000,
                             y_axis='linear')
    plt.title('STFT Amplitude')
    plt.colorbar(format='%+0.05f')
    #plt.colorbar(format='%+2.0f').set_label('[dB]')
    plt.ylim(0, 20000)
    plt.tight_layout()

    plt.subplot(2,2,2)
    librosa.display.specshow(amp_normal_db,
                             sr=96000,
                             hop_length=96000,
                             y_axis='linear')
    plt.title('STFT Amplitude in dB')
    plt.colorbar(format='%+2.0f').set_label('[dB]')
    plt.ylim(0, 20000)
    plt.tight_layout()

    plt.subplot(2,2,3)
    librosa.display.specshow(amp_normal_db_max,
                             sr=96000,
                             hop_length=96000,
                             y_axis='linear')
    plt.title('STFT Amplitude in dB (max)')
    plt.colorbar(format='%+2.0f').set_label('[dB]')
    plt.ylim(0, 20000)
    plt.tight_layout()

    plt.subplot(2,2,4)
    librosa.display.specshow(amp_normal_db_max2,
                             sr=96000,
                             hop_length=96000,
                             y_axis='linear')
    plt.title('STFT Amplitude in dB (max) without normalization')
    plt.colorbar(format='%+2.0f').set_label('[dB]')
    plt.ylim(0, 20000)
    plt.tight_layout()
    plt.show()



