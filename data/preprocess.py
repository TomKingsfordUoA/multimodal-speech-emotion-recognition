import os
import csv

import pandas as pd
import numpy as np
import librosa
import tqdm


def load_audio_file(path):
    audio_data, sample_rate = librosa.load(path)
    return audio_data, sample_rate


def extract_features(audio_data, sample_rate):
    sig_mean = np.mean(abs(audio_data))
    sig_std = np.std(audio_data)

    rmse = librosa.feature.rms(audio_data + 0.0001)[0]

    audio_data_harmonic = librosa.effects.hpss(audio_data)[0]

    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))

    # based on the pitch detection algorithm mentioned here:
    # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
    cl = 0.45 * sig_mean
    center_clipped = []
    for s in audio_data:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))

    return {
        'sig_mean': sig_mean,
        'sig_std': sig_std,
        'rmse_mean': np.mean(rmse),
        'rmse_std': np.std(rmse),
        'silence': silence,
        'harmonic': 1e3 * np.mean(audio_data_harmonic),
        'auto_corr_max': 1e3 * np.max(auto_corrs)/len(auto_corrs),
        'auto_corr_std': np.std(auto_corrs),
    }


def featurize_dataframe(df):
    features = []
    for idx, path in tqdm.tqdm(df['path'].items(), total=df.shape[0]):
        audio_data, sample_rate = load_audio_file(os.path.join(os.path.dirname(__file__), '..', path))
        features.append(extract_features(audio_data=audio_data, sample_rate=sample_rate))

    return df.join(pd.DataFrame(data=features, index=df.index))



def main():
    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_ravdess.csv'))
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_ravdess.csv'))
    print(df_train.shape)
    print(df_train.columns)
    print(df_test.shape)
    print(df_test.columns)

    df_train = featurize_dataframe(df_train)
    df_test = featurize_dataframe(df_test)

    df_train.to_csv(os.path.join(os.path.dirname(__file__), 'train_features.csv'), index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
    df_test.to_csv(os.path.join(os.path.dirname(__file__), 'test_features.csv'), index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()

