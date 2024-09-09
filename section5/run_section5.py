import argparse
import os
import sys

# enable import from another directory
sys.path.insert(0, os.getcwd())

from co_training_classifier import CoTrainingClassifier
from section3.knn import kNN
from tools.utils import load_csv_data, train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 5 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Define Dataset Path', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (co_train, self_train, pseudolabel)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)

    args = parser.parse_args()

    # load dataset (We use dummy data first)
    x_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)','Generosity', 'Dystopia Residual']
    y_columns = 'Region'
    x_data, y_data = load_csv_data(args.dataset, x_columns, y_columns)
    # divide data
    n_samples, n_features = x_data.shape
    y_data[:n_samples//2] = -1

    x_test = x_data[-n_samples//4:]
    y_test = y_data[-n_samples//4:]

    x_labeled = x_data[n_samples//2:n_samples//4]
    y_labeled = y_data[n_samples//2:n_samples//4]
    y_data = y_data[:-n_samples//4]
    x_data = x_data[:-n_samples//4]

    x_1 = x_data[:,:n_features // 2]
    x_2 = x_data[:, n_features // 2:]
    # x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)

    if args.mode == 'predict':
        if args.algo == 'co_train':
            co_preds = CoTrainingClassifier(kNN())
            co_preds.fit(x_1, x_2, y_data)
            y_pred = co_preds.predict(x_test[:, :n_features // 2], x_test[:, n_features // 2:])
            print(y_pred)
