import numpy as np
import os
import pandas as pd
import sys

# enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.utils import calc_distance

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 4, dist_metric: str = 'euclid', p: int = 3):
        self.n_clusters = n_clusters
        self.dist_metric = dist_metric
        self.p = p
    
    def fit_predict(self, x_data: np.ndarray) -> np.array:
        self.labels = np.arange(len(x_data))
        self.centroids = [[x] for x in x_data]
        print(len(self.centroids))
        for _ in range(self.n_clusters):
            min_dist = np.inf
            min_dist_idx = []

            for i in range(len(self.centroids) - 1):
                for j in range(i + 1, len(self.centroids)):
                    dist = self.cluster_distance(self.centroids[i], self.centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_idx = (i,j)
            self.labels[np.equal(self.labels, min_dist_idx[1])] = min_dist_idx[0]
            self.labels[np.greater(self.labels, min_dist_idx[1])] -= 1
            self.centroids[min_dist_idx[0]].extend(self.centroids.pop(min_dist_idx[1]))
        return np.array(self.labels)
    
    def cluster_distance(self, cluster_1: list, cluster_2: list):
        cluster_1_mean = np.mean(cluster_1, axis=0)
        cluster_2_mean = np.mean(cluster_2, axis=0)
        calc_dist = calc_distance(cluster_1_mean, cluster_2_mean, self.dist_metric, self.p)
        return (len(cluster_1) * len(cluster_2)) / (len(cluster_1) + len(cluster_2)) * calc_dist

