import gdown
import os
import pandas as pd

# Datasets URL
WHR_DATASETS_URL = 'https://drive.google.com/uc?id=1NLS0sEBWYjY-tHmnndUYqtefz9R3Clrv' # for classification
SHOP_DATASETS_URL = 'https://drive.google.com/uc?id=1ypjmMFDTmIoYr3x9WT9Pkw0Aa2b370Fc' # for clustering

# download whr_datasets
def download_datasets():
    #define path
    dataset_path = os.getcwd() + '/dataset'
    whr_file_path = os.path.join(dataset_path, 'temp_dataset_whr.csv')
    shop_data_path = os.path.join(dataset_path, 'shop_customer_dataset.csv')
    # download and save dataset
    gdown.download(url=WHR_DATASETS_URL, output=whr_file_path, quiet=False)
    gdown.download(url=SHOP_DATASETS_URL, output=shop_data_path, quiet=False)

def load_dataset(dataset_path: str):
    # read csv dataset
    df = pd.read_csv(dataset_path)
    print(df.head(5))

def sort_dataset(dataset_path: str, sort_type: str, column_name: str):
    # read csv dataset
    df = pd.read_csv(dataset_path)
    if sort_type == 'asc': # if ascending
        print(df.sort_values(by=[column_name], ascending=True))
    else: # if descending
        print(df.sort_values(by=[column_name], ascending=False))

def simplify_whr_classes(dataset_path: str):
    '''
        We try to simplify region name so it can contains not so many classes
    '''
    df = pd.read_csv(dataset_path, sep=',', index_col=False)
    temp_df = df.copy()
    for i in range(0, len(df['Region'])):
        if 'Asia' in df['Region'][i]:
            temp_df['Region'].replace(df['Region'][i], 'Asia', inplace=True)
        elif 'Africa' in df['Region'][i]:
            temp_df['Region'].replace(df['Region'][i], 'Africa', inplace=True)
        elif 'America' in df['Region'][i]:
            temp_df['Region'].replace(df['Region'][i], 'America', inplace=True)
        elif 'Australia' in df['Region'][i]:
            temp_df['Region'].replace(df['Region'][i], 'Asia', inplace=True)
        elif 'Europe' in df['Region'][i]:
            temp_df['Region'].replace(df['Region'][i], 'Europe', inplace=True)
    # save new dataset
    temp_df.to_csv('./dataset/whr_dataset.csv', index=False)