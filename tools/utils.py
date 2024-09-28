import numpy as np
import pandas as pd

from collections import Counter

def calc_distance(vec1: np.ndarray, vec2: np.ndarray, type: str = 'euclid', p: int = 3):
    '''
        Function for calculate distance between two vectors. Notes: if p = 1, equal to euclidean distance; if p = 2, equal to manhattan distance

        Input: vector 1, vector 2, type of disctance (default is euclidean distance), constant for minkowski distance
        
        Output: result of distance in floating point
    '''
    if type == 'euclid':
        return np.sqrt(np.sum((np.subtract(vec1, vec2))**2))
    elif type == 'manhattan':
        return np.sum(np.abs(np.subtract(vec1, vec2)))
    elif type ==  'minkowski':
        return np.sum(np.abs(np.subtract(vec1, vec2))**p) ** (1/p)

def calc_linkage(linkage_type: str, dist_result: float):
    if linkage_type == 'complete': # complete linkage
        return np.max(dist_result)
    elif linkage_type == 'single': # single linkage
        return np.min(dist_result)

def load_csv_data(dataset_path: str, x_columns: list, y_columns: str) -> tuple:
    ''' 
        Function for loading dataset from csv file, Only read several columns need to process

        Input: dataset path, list of columns to process

        Output: x,y in tuple
    '''
    df = pd.read_csv(dataset_path, sep=',', index_col=False) # read dataset first
    classes = df[y_columns].to_list() # we use region as classes
    classes = classes_to_int(classes)
    # use dataset based on new columns
    new_df = df[x_columns]
    # fill data
    new_df = new_df.mask(new_df==0).fillna(round(new_df.mean(),5))
    dataset = np.array(new_df, dtype=float)
    class_data = np.array(classes, dtype=int)
    # data normalization
    # dataset_normalization(dataset)
    x_data = dataset
    y_data = class_data
    return x_data, y_data

def classes_to_int(category_list: list) -> list:
    '''
        Function to turn list of string category into integer
        
        Input: list of classes

        Output: list of classes in integer
    '''
    new_category_list = []
    temp_filter = [n for n,v in Counter(category_list).items() if v >= 1]
    for i in range(0, len(category_list)):
        for j in range(0, len(temp_filter)):
            if category_list[i] == temp_filter[j]:
                new_category_list.append(j)
    return new_category_list

def train_test_split(x_data: np.ndarray, y_data: np.ndarray, train_size: float = 0.7):
    '''
        Function to split data into training and testing

        Input: x data, y data, training size
        
        Output: x train, y train, x test, y test
    '''
    temp_arr = np.random.rand(x_data.shape[0])
    split = temp_arr < np.percentile(temp_arr, train_size * 100)
    x_train = x_data[split]
    y_train = y_data[split]
    x_test = x_data[~split]
    y_test = y_data[~split]
    return x_train, y_train, x_test, y_test

def one_hot_label(y_data: np.ndarray, n_col=None) -> np.ndarray:
    '''
        Function to turn class into one hot encode labelling
    '''
    if not n_col:
        n_col = np.amax(y_data) + 1
    one_hot = np.zeros((y_data.shape[0], n_col))
    one_hot[np.arange(y_data.shape[0]), y_data] = 1
    return one_hot