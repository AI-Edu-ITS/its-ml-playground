import gdown
import os
import pandas as pd

# Datasets URL
WHR_DATASETS_URL = 'https://drive.google.com/uc?id=1NLS0sEBWYjY-tHmnndUYqtefz9R3Clrv'

# download whr_datasets
def download_whr_datasets():
    #define path
    dataset_path = os.getcwd() + '/dataset'
    file_path = os.path.join(dataset_path, 'temp_dataset.csv')
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

def simplify_classes(dataset_path: str):
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