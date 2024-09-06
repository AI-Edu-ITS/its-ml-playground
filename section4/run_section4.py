import argparse
import os
import sys

# Enable import from another directory
sys.path.insert(0, os.getcwd())

from ann import MLPClassifier
from logistic_regression import LogisticRegression, MultiLogisticRegression, visualize_logits_loss
from tools.classification_metrics import evaluation_report
from tools.utils import load_csv_data, train_test_split

from sklearn.linear_model import LogisticRegression as LR

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 4 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Define Dataset Path', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (logistic, svm, ann)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)
    parser.add_argument('-v', '--verbose', help='Define is log printed in console or not', default=False, type=bool)

    # logistic regression args
    parser.add_argument('-lr', '--learning_rate', help='Define learning rate for Logistic Regression', default=0.01, type=float)
    parser.add_argument('-i', '--iter', help='Define number of iteration for Logistic Regression', default=100, type=int)
    parser.add_argument('-eps', '--epsilon', help='Define number of error rate threshold for Logistic Regression algorithm', default=0.9, type=float)

    args = parser.parse_args()

    #define dataset
    # You can always change which columns you want to run (choose at least 2 columns with 1 column as y value)
    x_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)','Generosity', 'Dystopia Residual']
    y_columns = 'Region'
    x_data, y_data = load_csv_data(args.dataset, x_columns, y_columns)
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)

    if args.mode == 'predict':
        # WARNING: USE LOGISTIC REGRESSION ONLY FOR 2 CLASS PREDICTION!!!
        if args.algo == 'logistic':
            logits_preds = LogisticRegression(args.learning_rate, args.iter, args.epsilon, args.verbose)
            logits_preds.fit(x_train, y_train)
            result = logits_preds.predict(x_test)
        elif args.algo == 'multi_logistic':
            multi_logits_preds = MultiLogisticRegression(args.learning_rate, args.iter, verbose=args.verbose)
            multi_logits_preds.fit(x_train, y_train)
            result = multi_logits_preds.predict(x_test)
            print(multi_logits_preds.loss_list)
        elif args.algo == 'ann':
            ann_preds = MLPClassifier()
        evaluation_report(args.algo, result, y_test)
    elif args.mode =='vis':
        if args.algo == 'logistic' or args.algo == 'multi_logistic':
            multi_logits_preds = MultiLogisticRegression(args.learning_rate, args.iter)
            multi_logits_preds.fit(x_train, y_train)
            visualize_logits_loss(multi_logits_preds.loss_list)