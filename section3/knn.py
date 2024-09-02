import numpy as np

from utils_section3 import calc_distance

class kNN():
    def __init__(self, k_neighbours=3, dist_metric='euclid', p=3):
        self.k_neighbours = k_neighbours
        self.dist_metric = dist_metric
        self.p = p
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
    
    def get_neighbours(self, test_data) -> list:
        dist = []
        neighbours = []
        for (train_data, train_class) in zip(self.x_train, self.y_train):
            temp_dist = calc_distance(train_data, test_data, self.dist_metric, self.p)
            dist.append((temp_dist, train_class))
        dist.sort(key=lambda x: x[0])
        for i in range(self.k_neighbours):
            neighbours.append(dist[i][1])
        return neighbours

    def predict(self, x_test: np.ndarray):
        preds = []
        for test_data in x_test:
            nearest = self.get_neighbours(test_data)
            major = max(set(nearest), key=nearest.count)
            preds.append(major)
        return np.array(preds)

# import numpy as np

# from sklearn.model_selection import train_test_split

# from utils_section3 import load_csv_data

# '''
#     Function to get all nearest neighbours based on training data and its class

#     Input: list of train data, list of train class, number of k
#     Output: list of neighbours
# '''
# def get_neighbours(x_train: np.ndarray, y_train: np.ndarray, num_neighbors: int) -> list:
#     dist = []
#     neighbours = []
#     for (train_data, train_class) in zip(x_train, y_train):
#         temp_dist = calc_distance(train_data, train_class)
#         dist.append((temp_dist, train_class))
#     dist.sort(key=lambda x: x[0])
#     for i in range(num_neighbors):
#         neighbours.append(dist[i][1])
#     return neighbours

# def preds_knn(x_test: np.ndarray):
#     preds = []
#     for test_data in x_test:
#         neighbours = get_neighbours(test_data)

# '''
#     Function to implement knn algorithm
# '''
# def knn(dataset_path: str, train_size: float, num_neighbors: int) -> list:
#     # You can always change which columns you want to run (choose at least 2 columns with 1 column as y value)
#     process_columns = ['Family','Health (Life Expectancy)','Economy (GDP per Capita)','Happiness Score']
#     x_data, y_data, class_data = load_csv_data(dataset_path, process_columns)
#     print(class_data)
#     # split dataset
#     x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, test_size=0.3)
#     print(type(x_train))



# #     # # compare first
# #     # x_train, x_test, y_train, y_test = tr(x_data, y_data, test_size=0.3)
# #     # x_train_1, y_train_1, x_test_1, y_test_1 = train_test_split(x_data, y_data, 0.7)
# #     # print(f'sklearn split = {len(x_train)}, {len(y_train)}, {len(x_test)}, {len(y_test)}')
# #     # print(f'from sratch split = {len(x_train_1)}, {len(y_train_1)}, {len(x_test_1)}, {len(y_test_1)}')