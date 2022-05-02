# Code for making ML classifiers' input vector
# Two classes for classifiers are 0 (media) and 1 (natural)
from joblib import load
import librosa
import librosa.display
import numpy as np
import pandas as pd
import scipy

#enter path to saved .wav file (record at 16khz)
filename = 'replace this with path to file'

# All code below is from librosa to extract features from the audio
def create_chroma_df(chroma):
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    chroma_df = pd.DataFrame()
    for i in range(0, 12):
        chroma_df['chroma ' + str(i) + ' mean'] = chroma_mean[i]
        chroma_df['chroma ' + str(i) + ' std'] = chroma_std[i]
    chroma_df.loc[0] = np.concatenate((chroma_mean, chroma_std), axis=0)

    return chroma_df

def create_mfccs_df(mfccs):
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    mfccs_df = pd.DataFrame()
    for i in range(0, 13):
        mfccs_df['mfccs ' + str(i) + ' mean'] = mfccs_mean[i]
        mfccs_df['mfccs ' + str(i) + ' std'] = mfccs_std[i]
    mfccs_df.loc[0] = np.concatenate((mfccs_mean, mfccs_std), axis=0)

    return mfccs_df

def create_rms_df(rms):
    # rms_mean = np.mean(rms)
    rms_range = np.ptp(rms)
    rms_std = np.std(rms)
    rms_skew = scipy.stats.skew(rms, axis=1)[0]

    rms_df = pd.DataFrame()
    # rms_df['rms mean'] = 0
    rms_df['rms range'] = 0
    rms_df['rms std'] = 0
    rms_df['rms skew'] = 0
    rms_df.loc[0] = [
        # rms_mean,
        rms_range, rms_std, rms_skew]
    return rms_df


def create_spectral_df(cent, contrast, rolloff, flatness, bandwidth):
    # spectral centroids values
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_skew = scipy.stats.skew(cent, axis=1)[0]

    # spectral contrasts values
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # spectral rolloff points values
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    rolloff_skew = scipy.stats.skew(rolloff, axis=1)[0]

    # spectral flatness values
    flat_mean = np.mean(flatness)
    flat_std = np.std(flatness)
    flat_skew = scipy.stats.skew(flatness, axis=1)[0]

    # bandwidth values
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)
    bandwidth_skew = scipy.stats.skew(bandwidth, axis=1)[0]

    spectral_df = pd.DataFrame()
    collist = ['cent mean', 'cent std', 'cent skew',
               'flat mean', 'flat std', 'flat skew',
               'rolloff mean', 'rolloff std', 'rolloff skew',
               'bandwidth mean', 'bandwidth std', 'bandwidth skew']
    for i in range(0, 7):
        collist.append('contrast ' + str(i) + ' mean')
        collist.append('contrast ' + str(i) + ' std')

    for c in collist:
        spectral_df[c] = 0
    data = np.concatenate((
        [cent_mean, cent_std, cent_skew],
        [flat_mean, flat_std, flat_skew],
        [rolloff_mean, rolloff_std, rolloff_skew],
        [bandwidth_mean, bandwidth_std, bandwidth_skew],
        contrast_mean, contrast_std),
        axis=0)
    spectral_df.loc[0] = data

    return spectral_df

def create_zrate_df(zrate):
    zrate_mean = np.mean(zrate)
    zrate_std = np.std(zrate)
    zrate_skew = scipy.stats.skew(zrate, axis=1)[0]

    zrate_df = pd.DataFrame()
    zrate_df['zrate mean'] = 0
    zrate_df['zrate std'] = 0
    zrate_df['zrate skew'] = 0
    zrate_df.loc[0] = [zrate_mean, zrate_std, zrate_skew]

    return zrate_df

def create_beat_df(tempo):
    beat_df = pd.DataFrame()
    beat_df['tempo'] = tempo
    beat_df.loc[0] = tempo
    return beat_df


# Create a function that consolidates all of the above feature extraction functions
def extract_features(audio):

    y, sr = librosa.load(audio, sr=16000, dtype=np.float32)  # read the wav files

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_time_diff = np.ediff1d(beat_times)
    beat_nums = np.arange(1, np.size(beat_times))

    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)

    rms = librosa.feature.rms(y=y)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    flatness = librosa.feature.spectral_flatness(y=y)

    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    zrate = librosa.feature.zero_crossing_rate(y_harmonic)

    chroma_df = create_chroma_df(chroma)

    mfccs_df = create_mfccs_df(mfccs)

    rms_df = create_rms_df(rms)

    spectral_df = create_spectral_df(cent, contrast, rolloff, flatness, bandwidth)

    zrate_df = create_zrate_df(zrate)

    beat_df = create_beat_df(tempo)

    final_df = pd.concat((chroma_df,
                          mfccs_df, rms_df, spectral_df,
                          zrate_df, beat_df
                          ), axis=1)
    return final_df

#load the scaler
scaler = load('scaler.save')

#extract audio features
feature_vector = extract_features(filename)

#scale the features
feature_vector = scaler.transform(feature_vector)

#load the respective classifier (example below)
svc = load('SVC.joblib')

#make the corresponding prediction using classifer.predict(feature_vector) -- example below
print(svc.predict(feature_vector))