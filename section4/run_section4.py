import argparse
import numpy as np
import os
import sys

# Enable import from another directory
sys.path.insert(0, os.getcwd())

from ann import MLPClassifier, visualize_loss
from logistic_regression import LogisticRegression, MultiLogisticRegression, visualize_logits_loss
from svm import SVMNoKernel, visualize_svm_result
from tools.classification_metrics import evaluation_report
from tools.utils import load_csv_data, train_test_split, one_hot_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 4 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (logistic, multi_logistic, svm, ann)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.8, type=float)
    parser.add_argument('-v', '--verbose', help='Define is log printed in console or not', default=False, type=bool)
    parser.add_argument('-it', '--iter', help='Define number of iteration or epoch', default=100, type=int)
    parser.add_argument('-bs', '--batch_size', help='Define batch size for process', default=32, type=int)
    parser.add_argument('-rs', '--random_seed', help='Define random seed for initialization', default=42, type=int)

    # logistic regression args
    parser.add_argument('-lr', '--learning_rate', help='Define learning rate for Logistic Regression', default=0.01, type=float)
    parser.add_argument('-eps', '--epsilon', help='Define number of error rate threshold for Logistic Regression algorithm', default=0.9, type=float)

    # svm args
    parser.add_argument('-rg', '--regularization', help='Define regularization', default=0.01, type=float)

    # ann args
    parser.add_argument('-hl', '--hidden_layer', help='Define hidden layer of the network', default=4, type=int)
    parser.add_argument('-ac', '--activation', help='Define activation in mlp (sigmoid, relu, tanh, softmax)', default='relu', type=str)

    args = parser.parse_args()

    #define dataset
    # You can always change which columns you want to run (choose at least 2 columns with 1 column as y value)
    if args.algo == 'svm' or args.algo == 'logistic':
        dataset_path = './shop_customer_dataset.csv'
        x_columns = ['Annual Income ($)','Spending Score (1-100)','Work Experience','Family Size']
        y_columns = 'Gender'
    else:
        dataset_path = './dataset/whr_dataset.csv'
        x_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)','Generosity']
        y_columns = 'Region'
    x_data, y_data = load_csv_data(dataset_path, x_columns, y_columns)
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)

    if args.mode == 'predict':
        # WARNING: USE LOGISTIC REGRESSION ONLY FOR 2 CLASS PREDICTION!!!
        if args.algo == 'logistic':
            logits_preds = LogisticRegression(args.learning_rate, args.iter, args.epsilon, args.verbose)
            logits_preds.fit(x_train, y_train)
            result = logits_preds.predict(x_test)
        elif args.algo == 'multi_logistic':
            multi_logits_preds = MultiLogisticRegression(args.learning_rate, args.iter, args.batch_size, args.random_seed, args.verbose)
            multi_logits_preds.fit(x_train, y_train)
            result = multi_logits_preds.predict(x_test)
        elif args.algo == 'ann':
            input_size = x_train.shape[1]
            output_size = len(np.unique(y_train))
            ann_preds = MLPClassifier(input_size, args.hidden_layer, output_size, args.learning_rate, args.iter, args.activation, args.verbose)
            ann_preds.fit(x_train, y_train)
            result = ann_preds.predict(x_test)
        elif args.algo == 'svm':
            svm_preds = SVMNoKernel(args.iter, args.learning_rate, args.regularization)
            svm_preds.fit(x_train, y_train)
            result = svm_preds.predict(x_test)
            # sklearn
            from sklearn.svm import SVC
            skmodel = SVC()
            skmodel.fit(x_train, y_train)
            sk_predictions = skmodel.predict(x_test)
            evaluation_report(args.algo, sk_predictions, y_test)
        evaluation_report(args.algo, result, y_test)
    elif args.mode =='vis':
        if args.algo == 'logistic' or args.algo == 'multi_logistic':
            multi_logits_preds = MultiLogisticRegression(args.learning_rate, args.iter)
            multi_logits_preds.fit(x_train, y_train)
            visualize_logits_loss(multi_logits_preds.loss_list)
        elif args.algo == 'svm':
            svm_preds = SVMNoKernel(args.iter, args.learning_rate, args.regularization)
            svm_preds.fit(x_data, y_data)
            weight = svm_preds.weight
            bias = svm_preds.bias
            visualize_svm_result(x_data, y_data, weight, bias)
        elif args.algo == 'ann':
            input_size = x_train.shape[1]
            output_size = len(np.unique(y_train))
            ann_preds = MLPClassifier(input_size, args.hidden_size, output_size, args.learning_rate, args.iter, -1, -1, args.activation)
            ann_preds.fit(x_train, y_train)
            visualize_loss(ann_preds.epoch_list, ann_preds.error_list)