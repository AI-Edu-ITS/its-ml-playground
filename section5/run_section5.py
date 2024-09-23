import argparse
import numpy as np
import os
import sys

# enable import from another directory
sys.path.insert(0, os.getcwd())

from co_training_classifier import CoTrainingClassifier
from pseudolabelling import Pseudolabel
from section3.knn import kNN
from self_training_classifier import SelfTrainingClassifier
from tools.utils import load_csv_data, train_test_split
from tools.classification_metrics import evaluation_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 5 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (co_train, self_train, pseudolabel)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)
    parser.add_argument('-it', '--iter', help='Define number of iteration or epoch', default=10, type=int)

    # pseudo label args
    parser.add_argument('-sr', '--sample_rate', help='Define sample rate for pseudo label dataset', default=0.2, type=float)

    args = parser.parse_args()

    # load dataset (We use dummy data first)
    dataset_path = './dataset/abalone_dataset.csv'
    x_columns = ['Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']
    y_columns = 'Gender'
    x_data, y_data = load_csv_data(dataset_path, x_columns, y_columns)

    if args.mode == 'predict':
        if args.algo == 'co_train':
            # divide data
            n_samples, n_features = x_data.shape
            y_data[:n_samples//2] = -1

            x_test = x_data[-n_samples//4:]
            y_test = y_data[-n_samples//4:]

            y_data = y_data[:-n_samples//4]
            x_data = x_data[:-n_samples//4]

            x_1: np.ndarray = x_data[:,:n_features // 2]
            x_2: np.ndarray = x_data[:, n_features // 2:]

            co_preds = CoTrainingClassifier(kNN(), args.iter)
            co_preds.fit(x_1, x_2, y_data)
            preds = co_preds.predict(x_test[:, :n_features // 2], x_test[:, n_features // 2:])
        elif args.algo == 'pseudolabel':
            # split dataset
            x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)
            pseudo_preds = Pseudolabel(kNN(), args.sample_rate)
            pseudo_preds.fit(x_train, y_train, x_test, y_test)
            preds = pseudo_preds.predict(x_test)
        elif args.algo == 'self_train':
            self_preds = SelfTrainingClassifier()
        evaluation_report(args.algo, preds, y_test)
