import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import requests

from random import randrange

# define dataset path
iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# convert to float

# convert to integer
def convert_to_int() -> dict:
    return {}

# split dataset based on its n folds number
def cross_val_split(dataset: list, n_folds: int):
    data_split = []
    temp_data = dataset
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        temp_fold = []
        while len(temp_fold) < fold_size:
            index = randrange(len(temp_data))
            temp_fold.append(temp_data.pop(index))
    return 0

# find minimum and maximum value from each vector
# def dataset_minmax(dataset: list) -> list:


# download dataset
def download_save_data(args):
    dataset_path = os.getcwd() + '/dataset'
    if args.dataset == 'iris':
        response = requests.get(iris_url)
        file_path = os.path.join(dataset_path, 'iris.csv')

    # download and save dataset
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print('Dataset Successfully Downloaded!')
    else:
        print('Failed To Download File')

# turn csv to list
def load_csv(csv_path: str) -> list:
    dataset = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
    for row in csv_reader:
        if not row:
            continue
        else:
            dataset.append(row)
    return dataset

# def vis_data(args):
#     return 0

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Dataset Parser')
#     parser.add_argument('command', help='Choose between download or visualize')
#     parser.add_argument('--dataset', help='Choose Dataset between Iris, etc.')
#     args = parser.parse_args()
#     if args.command == 'download':
#         download_save_data(args)
#     elif args.command == 'vis':
#         vis_data(args)
#     else:
#         print('command not recognized')