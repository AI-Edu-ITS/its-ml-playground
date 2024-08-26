import argparse

from src.algo.knn import knn

def run_preds():
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='choose dataset name from iris, etc.')
    parser.add_argument('--algo', help='choose ML algorithm from knn, etc.')
    parser.add_argument('--folds', type=int, help='input number of group for data partition', default=5)
    args = parser.parse_args()
    run_preds(args)