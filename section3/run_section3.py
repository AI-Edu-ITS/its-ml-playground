import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from tools.classification_metrics import evaluation_report
from decision_tree import DecisionTreeClassifier
from knn import kNN, visualize_knn_best_k
from naive_bayes import GaussianNaiveBayes
from regression import SimpleLinearRegression, MultiLinearRegression, visualize_simple_regression, visualize_multi_regression
from tools.utils import load_csv_data, train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 3 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Define Dataset Path', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (simple_regression, multi_regression, knn, naive, tree)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)

    # linear regression specific
    parser.add_argument('-lr', '--learning_rate', help='Define learning rate for Linear Regression', default=0.0001, type=float)
    parser.add_argument('-eps', '--epsilon', help='Define number of error rate threshold for Multi Linear Regression algorithm', default=0.9, type=float)
    
    # knn specific
    parser.add_argument('-k', '--k_neighbours', help='Define number of k-neighbors for KNN algorithm', default=5, type=int)

    # decision tree specific
    parser.add_argument('-cr', '--criterion', help='Define criterion for decision tree (entropy, gini, log_loss)', default='gini', type=str)
    parser.add_argument('-md', '--max_depth', help='Define maximum depth for node tree in decision tree', default=5, type=int)

    # misc
    parser.add_argument('-dm', '--dist_metric', help='Distance metric used (euclid, manhattan, minkowski)', type=str, default='euclid')
    parser.add_argument('-p', '--p_value', help='Determine p value for minkowski distance', default=3, type=int)

    args = parser.parse_args()

    #define dataset
    # You can always change which columns you want to run (choose at least 2 columns with 1 column as y value)
    # Keep in mind that simple linear regression only use one feature to process
    if args.algo != 'simple_regression':
        x_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)','Generosity']
    else:
        x_columns = ['Freedom'] 
    y_columns = 'Region'
    x_data, y_data = load_csv_data(args.dataset, x_columns, y_columns)
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)
    
    # choose prediction
    if args.mode == 'predict':
        if args.algo == 'simple_regression': # for Simple Linear Regression algorithm
            lr_preds = SimpleLinearRegression()
            lr_preds.fit(x_train, y_train)
            result = lr_preds.predict(x_test)
        elif args.algo == 'multi_regression': # for Multi Linear Regression algorithm
            mlr_preds = MultiLinearRegression(args.learning_rate, args.epsilon)
            mlr_preds.fit(x_train, y_train)
            result = mlr_preds.predict(x_test)
        elif args.algo == 'knn': # for K-Nearest Neighbor algorithm
            knn_preds = kNN(args.k_neighbours, args.dist_metric)
            knn_preds.fit(x_train, y_train)
            result = knn_preds.predict(x_test)
            result_proba = knn_preds.predict_proba(x_test)
            print(result_proba)
        elif args.algo == 'naive': # for Naïve Bayes algorithm
            naive_preds = GaussianNaiveBayes()
            naive_preds.fit(x_train, y_train)
            result = naive_preds.predict(x_test)
        elif args.algo == 'tree': # for Decision Tree algorithm
            tree_preds = DecisionTreeClassifier(args.criterion, args.max_depth)
            tree_preds.fit(x_train, y_train)
            result = tree_preds.predict(x_test)
        evaluation_report(args.algo, result, y_test)
    # choose visualize
    elif args.mode == 'vis':
        if args.algo == 'simple_regression':
            visualize_simple_regression(x_train, y_train, x_test, y_test)
        elif args.algo == 'multi_regression':
            visualize_multi_regression(x_train, y_train, x_test, y_test)
        elif args.algo == 'knn':
            visualize_knn_best_k(x_train, y_train, x_test, y_test, args.dist_metric, args.p_value)