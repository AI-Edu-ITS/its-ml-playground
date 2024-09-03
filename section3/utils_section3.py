import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import Counter

# # normalize dataset
# def dataset_normalization(dataset: np.ndarray):
#     for i in range(0, dataset.shape[1]-1):
#         dataset[:,i] = ((dataset[:,i] - np.mean(dataset[:,i]))/np.std(dataset[:,i]))

'''
    Function for calculate distance between two vectors
    Notes: if p = 1, equal to euclidean distance; if p = 2, equal to manhattan distance

    Input: vector 1, vector 2, type of disctance (default is euclidean distance), constant for minkowski distance
    Output: result of distance in floating point
'''
def calc_distance(vec1: list, vec2: list, type: str = 'euclid', p: int = 3) -> float:
    if type == 'euclid':
        return np.sqrt(np.sum((vec1 - vec2)**2))
    elif type == 'manhattan':
        return np.sum(np.abs(vec1 - vec2))
    elif type ==  'minkowski':
        return np.sum(np.abs(vec1 - vec2)**p) ** (1/p)

''' 
    Function for loading dataset from csv file, Only read several columns need to process

    Input: dataset path, list of columns to process
    Output: x,y in tuple
'''
def load_csv_data(dataset_path: str, process_columns: list) -> tuple:
    x_columns = len(process_columns) - 1
    df = pd.read_csv(dataset_path, sep=',', index_col=False) # read dataset first
    classes = df['Region'].to_list() # we use region as classes
    classes = classes_to_int(classes)
    # use dataset based on new columns
    new_df = df[process_columns]
    # fill data
    new_df = new_df.mask(new_df==0).fillna(round(new_df.mean(),5))
    dataset = np.array(new_df, dtype=float)
    class_data = np.array(classes, dtype=int)
    # data normalization
    # dataset_normalization(dataset)
    x_data = dataset[:,:x_columns]
    y_data = class_data
    return x_data, y_data

'''
    Function to turn list of string category into integer
    
    Input: list of classes
    Output: list of classes in integer
'''
def classes_to_int(category_list: list) -> list:
    new_category_list = []
    temp_filter = [n for n,v in Counter(category_list).items() if v >= 1]
    for i in range(0, len(category_list)):
        for j in range(0, len(temp_filter)):
            if category_list[i] == temp_filter[j]:
                new_category_list.append(j)
    return new_category_list

'''
    Function to split data into training and testing

    Input: x data, y data, training size
    Output: x train, y train, x test, y test
'''

def train_test_split(x_data: np.ndarray, y_data: np.ndarray, train_size: float = 0.7) -> tuple:
    temp_arr = np.random.rand(x_data.shape[0])
    split = temp_arr < np.percentile(temp_arr, train_size * 100)
    x_train = x_data[split]
    y_train = y_data[split]
    x_test = x_data[~split]
    y_test = y_data[~split]
    return x_train, y_train, x_test, y_test
