import argparse
import numpy as np

from eval_supervised import evaluation_report
from knn import kNN
from regression import linear_regression
from utils_section3 import load_csv_data, train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 3 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, eval, vis)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Define Dataset Path')
    parser.add_argument('-a', '--algo', help='Run specific algo (regression, knn, naive, tree)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)
    
    # knn specific
    parser.add_argument('-k', '--k_neighbours', help='Define number of k-neighbors for KNN algorithm', default=5, type=int)

    # misc
    parser.add_argument('-dm', '--dist_metric', help='Distance metric used (euclid, manhattan, minkowski)', type=str, default='euclid')

    args = parser.parse_args()
    
    if args.mode == 'predict':
        # You can always change which columns you want to run (choose at least 2 columns with 1 column as y value)
        x_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)','Generosity']
        x_data, y_data = load_csv_data(args.dataset, x_columns)
        x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)
        if args.algo == 'regression':
            linear_regression(args.dataset)
        elif args.algo == 'knn':
            knn_preds = kNN(args.k_neighbours, args.dist_metric)
            knn_preds.fit(x_train, y_train)
            result = knn_preds.predict(x_test)
            evaluation_report(args.algo, result, y_test)