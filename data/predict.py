import os
import pickle
import csv

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def print_metrics(y, y_pred_argmax, label):
    print('{} Set Accuracy =  {:.3f}'.format(label, accuracy_score(y, y_pred_argmax)))
    print('{} Set F-score =  {:.3f}'.format(label, f1_score(y, y_pred_argmax, average='macro')))
    print('{} Set Precision =  {:.3f}'.format(label, precision_score(y, y_pred_argmax, average='macro')))
    print('{} Set Recall =  {:.3f}'.format(label, recall_score(y, y_pred_argmax, average='macro')))


def main():
    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_features.csv'), index_col=0)
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_features.csv'), index_col=0)
    print(df_train.shape)
    print(df_train.columns)
    print(df_test.shape)
    print(df_test.columns)

    # Convert emotion labels from text to ordinal:
    emotion_dict = {emotion: idx for idx, emotion in enumerate(sorted({emotion for _, emotion in df_train['emotion'].items()}))}
    emotion_dict_reverse = {idx: emotion for emotion, idx in emotion_dict.items()}
    print(emotion_dict)

    x_train_raw = df_train.drop(columns=['emotion', 'path']).values
    y_train_raw = df_train['emotion'].map(emotion_dict).values
    index_train, index_validation, x_train, x_validation, y_train, y_validation = train_test_split(df_train.index, x_train_raw, y_train_raw, test_size=0.2, random_state=42)
    x_test = df_test.drop(columns=['emotion', 'path']).values
    y_test = df_test['emotion'].map(emotion_dict).values

    print(x_train.shape)
    print(y_train.shape)
    print(x_validation.shape)
    print(y_validation.shape)
    print(x_test.shape)
    print(y_test.shape)

    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(emotion_dict),
    )
    xgb_classifier.load_model('xgb.txt')

    y_train_pred = xgb_classifier.predict_proba(x_train)
    y_train_pred_argmax = np.argmax(y_train_pred, axis=-1)

    y_validation_pred = xgb_classifier.predict_proba(x_validation)
    y_validation_pred_argmax = np.argmax(y_validation_pred, axis=-1)

    y_test_pred = xgb_classifier.predict_proba(x_test)
    y_test_pred_argmax = np.argmax(y_test_pred, axis=-1)

    print(y_train_pred.shape)
    print(y_train_pred_argmax.shape)
    print(y_validation_pred.shape)
    print(y_validation_pred_argmax.shape)
    print(y_test_pred.shape)
    print(y_test_pred_argmax.shape)

    print_metrics(y_train, y_train_pred_argmax, 'train')
    print_metrics(y_validation, y_validation_pred_argmax, 'validation')
    print_metrics(y_test, y_test_pred_argmax, 'test')

    # TODO(TK): save to files
    df_train_raw = df_train.rename(columns={'emotion': 'target'})[['path', 'target']]
    df_test = df_test.rename(columns={'emotion': 'target'})[['path', 'target']]

    df_train = df_train_raw.join(pd.DataFrame(data=y_train_pred, columns=[f'pred_{emotion_dict_reverse[idx]}' for idx in range(len(emotion_dict))], index=index_train), how='inner')
    df_validation = df_train_raw.join(pd.DataFrame(data=y_validation_pred, columns=[f'pred_{emotion_dict_reverse[idx]}' for idx in range(len(emotion_dict))], index=index_validation), how='inner')
    df_test = df_test.join(pd.DataFrame(data=y_test_pred, columns=[f'pred_{emotion_dict_reverse[idx]}' for idx in range(len(emotion_dict))], index=df_test.index))

    df_train.to_csv(os.path.join(os.path.dirname(__file__), 'train_pred.csv'), index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
    df_validation.to_csv(os.path.join(os.path.dirname(__file__), 'validation_pred.csv'), index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)
    df_test.to_csv(os.path.join(os.path.dirname(__file__), 'test_pred.csv'), index=True, header=True, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()

