import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# enable import from another directory
sys.path.insert(0, os.getcwd())

from tools.classification_metrics import calc_error_rate, calc_accuracy, calc_auc_score_multi, calc_auc_score_binary
from tools.utils import calc_distance

class kNN():
    '''
        K-Nearest Neighbor (KNN) class. This class contains 3 methods which are the foundation of KNN, namely calculate distance,
        finding nearest neighbour, and predict the new data location with its class. In KNN, we need to define number of neighbors
        which represented as k_neighbours and which metric used to calculate the distance between data.
    '''
    def __init__(self, k_neighbours: int = 5, dist_metric: str ='euclid', p: int = 3):
        self.k_neighbours = k_neighbours
        self.dist_metric = dist_metric
        self.p = p
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(self.y_train)
        self.num_classes = len(self.classes)
    
    def get_neighbours(self, test_data: np.ndarray) -> list:
        dist = []
        neighbours = []
        for (train_data, train_class) in zip(self.x_train, self.y_train):
            temp_dist = calc_distance(train_data, test_data, self.dist_metric, self.p)
            dist.append((temp_dist, train_class))
        dist.sort(key=lambda x: x[0])
        for i in range(self.k_neighbours):
            neighbours.append(dist[i][1])
        return neighbours

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(x_test), dtype=np.uint32)
        for idx, test_data in enumerate(x_test):
            nearest = self.get_neighbours(test_data)
            major = max(set(nearest), key=nearest.count)
            preds[idx] = major
        return preds
    
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        preds_proba = np.zeros((len(x_test), self.num_classes))
        for x_idx, test_data in enumerate(x_test):
            nearest = self.get_neighbours(test_data)
            for class_idx, class_label in enumerate(self.classes):
                preds_proba[x_idx, class_idx] = np.mean(np.equal(nearest, class_label))
        return preds_proba

def visualize_knn_best_k(    
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        dist_metric: str,
        p: int
    ):
    '''
        Function for visualize performance of knn. we try to find the best value of k in terms of accuracy and error rate
    '''
    error_list = []
    acc_list = []
    for i in range(1,40):
        knn = kNN(i, dist_metric, p)
        knn.fit(x_train, y_train)
        preds = knn.predict(x_test)
        error_list.append(calc_error_rate(preds, y_test))
        acc_list.append(calc_accuracy(preds,y_test))
    # plot graph error rate and acc rate
    _, (ax_acc, ax_err) = plt.subplots(2,1,constrained_layout=True)
    # plot acc
    ax_acc.plot(range(1,40),acc_list,color='red',linestyle='dashed',marker='o',markerfacecolor='green')
    ax_acc.set_title('Accuracy vs K-Val')
    plt.setp(ax_acc, xlabel='K-Val', ylabel='Accuracy')
    print(f'Best Accuracy = {max(acc_list)} when K = {acc_list.index(max(acc_list))}')
    # plot error
    ax_err.plot(range(1,40),error_list,color='red',linestyle='dashed',marker='o',markerfacecolor='blue')
    ax_err.set_title('Error Rate vs K-Val')
    plt.setp(ax_err, xlabel='K-Val', ylabel='Error Rate')
    print(f'Best Error Rate = {min(error_list)} when K = {error_list.index(min(error_list))}')
    plt.show()

def visualize_roc_auc_knn(preds_proba: np.ndarray, y_test: np.ndarray, y_data: np.ndarray):
    # auc_score, fpr_list, tpr_list = calc_auc_score_binary(preds_proba, y_test)
    num_class = np.unique(y_data)
    if len(num_class) <= 2:
        auc_score, fpr_list, tpr_list = calc_auc_score_binary(preds_proba, y_test)
    else:
        auc_score, fpr_list, tpr_list = calc_auc_score_multi(preds_proba, y_test, num_class)
    plt.plot([0,0,1], [0,1,1], color='red', label='Ideal ROC')
    plt.plot([0,1], [0,1], color='orange', label='Random ROC')
    plt.plot(tpr_list, fpr_list, color='green', label='Result ROC')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()