import os
import pickle

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
    print(df_train.shape)
    print(df_train.columns)

    # Convert emotion labels from text to ordinal:
    emotion_dict = {emotion: idx for idx, emotion in enumerate(sorted({emotion for _, emotion in df_train['emotion'].items()}))}
    print(emotion_dict)

    x_train_raw = df_train.drop(columns=['emotion', 'path']).values
    y_train_raw = df_train['emotion'].map(emotion_dict).values

    print(x_train_raw.shape)
    print(y_train_raw.shape)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=42)

    xgb_classifier = xgb.XGBClassifier(
        max_depth=7,
        learning_rate=1e-3,
        objective='multi:softprob',
        n_estimators=5000,
        sub_sample=0.8,
        num_class=len(emotion_dict),
        booster='gbtree',
        n_jobs=4,
    )

    xgb_classifier.fit(x_train, y_train)

    y_train_pred = xgb_classifier.predict_proba(x_train)
    y_train_pred_argmax = np.argmax(y_train_pred, axis=-1)

    y_validation_pred = xgb_classifier.predict_proba(x_validation)
    y_validation_pred_argmax = np.argmax(y_validation_pred, axis=-1)

    print_metrics(y_train, y_train_pred_argmax, 'train')
    print_metrics(y_validation, y_validation_pred_argmax, 'validation')

    pickle.dump(xgb_classifier, open('xgb.pickle', 'wb'))
    xgb_classifier.save_model('xgb.txt')
    xgb_classifier.save_model('xgb.json')


if __name__ == '__main__':
    main()

