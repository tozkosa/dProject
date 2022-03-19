import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import pandas as pd
DATA_ROOT = "../daon_data/ishikawa_test_small_wave"

def list_of_dirs(data_root):
    print("inside list of dirs")
    print(data_root)
    #print(os.listdir(data_root))
    file_name = []
    list2 = []
    list3 = []
    list4 = []
    for file in sorted(os.listdir(data_root)):
        # fname = file.split("_")
        # print(fname)
        print(file)
        file_name.append(file)
    data = {'fname': file_name}
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('test_ishikawa_wave.csv', encoding='utf-8')

def get_spectrum_magnitude(X, sr, f_ratio=0.5):
    f = np.linspace(0, sr, len(X[0]))
    f_bins = int(len(X[0]) * f_ratio)
    for i in range(len(X)):
        ft = np.fft.fft(X[i])
        ft_mag = np.absolute(ft)
        X[i] = ft_mag[:f_bins]
    return X


if __name__ == "__main__":
    print("Start!")
    #list_of_dirs(DATA_ROOT)
    X = []
    line = []
    no = []
    for i, file in enumerate(sorted(os.listdir(DATA_ROOT))):
        file_path = os.path.join(DATA_ROOT, file)
        #print(file_path)
        signal, sr = librosa.load(file_path)
        #print(f"{i}: {file}, sr={sr}, n_frames={len(signal)}")
        X.append(signal)
        fname = file.split("_")
        
        line.append(fname[0])
        no.append(fname[1])
        #print(fname[0], fname[1])

    print("get spectrum")
    X = get_spectrum_magnitude(X, sr, 0.5) 
    print(len(X))
    print(len(X[0]))

    """Principal Component Analysis"""
    print("PCA")
    pca = PCA(n_components=10)
    X_scaled = pca.fit_transform(X)

    """One Class SVM"""
    clf = OneClassSVM(gamma='scale').fit(X_scaled)
    print(clf.predict(X_scaled))
    Y = clf.score_samples(X_scaled)
    Y2 = clf.decision_function(X_scaled)
    print(len(Y2))
    print(Y2)

    data = {'line': line, 'number': no, 'score': Y}
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('test_ishikawa_wave.csv', encoding='utf-8')