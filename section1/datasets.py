import gdown
import os
import pandas as pd

# Datasets URL
WHR_DATASETS_URL = 'https://drive.google.com/uc?id=1NLS0sEBWYjY-tHmnndUYqtefz9R3Clrv'

# download whr_datasets
def download_whr_datasets():
    #define path
    dataset_path = os.getcwd() + '/dataset'
    file_path = os.path.join(dataset_path, 'whr_dataset.csv')
    # download and save dataset
    gdown.download(url=WHR_DATASETS_URL, output=file_path, quiet=False)

def load_whr_datasets(dataset_path: str):
    # read csv dataset
    df = pd.read_csv(dataset_path)
    print(df.head(5))

def sort_whr_datasets(dataset_path: str, sort_type: str, column_name: str):
    # read csv dataset
    df = pd.read_csv(dataset_path)
    if sort_type == 'asc': # if ascending
        print(df.sort_values(by=[column_name], ascending=True))
    else: # if descending
        print(df.sort_values(by=[column_name], ascending=False))

# import argparse
# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas
# import requests

# from random import randrange

# # define dataset path
# iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# # convert to float

# # convert to integer
# def convert_to_int() -> dict:
#     return {}

# # split dataset based on its n folds number
# def cross_val_split(dataset: list, n_folds: int):
#     data_split = []
#     temp_data = dataset
#     fold_size = int(len(dataset) / n_folds)
#     for _ in range(n_folds):
#         temp_fold = []
#         while len(temp_fold) < fold_size:
#             index = randrange(len(temp_data))
#             temp_fold.append(temp_data.pop(index))
#     return 0

# # find minimum and maximum value from each vector
# # def dataset_minmax(dataset: list) -> list:


# # download dataset
# def download_save_data(args):
#     dataset_path = os.getcwd() + '/dataset'
#     if args.dataset == 'iris':
#         response = requests.get(iris_url)
#         file_path = os.path.join(dataset_path, 'iris.csv')

#     # download and save dataset
#     if response.status_code == 200:
#         with open(file_path, 'wb') as file:
#             file.write(response.content)
#         print('Dataset Successfully Downloaded!')
#     else:
#         print('Failed To Download File')

# # turn csv to list
# def load_csv(csv_path: str) -> list:
#     dataset = []
#     with open(csv_path, 'r') as file:
#         csv_reader = csv.reader(file)
#     for row in csv_reader:
#         if not row:
#             continue
#         else:
#             dataset.append(row)s
#     return dataset

# # def vis_data(args):
# #     return 0

# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser('Dataset Parser')
# #     parser.add_argument('command', help='Choose between download or visualize')
# #     parser.add_argument('--dataset', help='Choose Dataset between Iris, etc.')
# #     args = parser.parse_args()
# #     if args.command == 'download':
# #         download_save_data(args)
# #     elif args.command == 'vis':
# #         vis_data(args)
# #     else:
# #         print('command not recognized')