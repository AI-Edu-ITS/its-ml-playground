import numpy as np
import os
import random
import sys

# enable import from another directory
sys.path.insert(0, os.getcwd())

from tools.utils import calc_distance

class KMeans():
    '''
        Class for running LLoyd Algorithm or known as K-Means Clustering. we used parameters inputs such as
        number of clusters, maximum iteration, and tolerance value for decalring model divergence. we also add distance
        metric to choose for calculate distance between centroid and p value for minkowski distance
    '''
    def __init__(self, n_clusters: int = 8, iterations: int = 300, dist_metric: str = 'euclid', p: int = 3) -> None:
        self.n_clusters = n_clusters
        self.iter = iterations
        self.dist_metric = dist_metric
        self.p = p
    
    def fit(self, x_train: np.ndarray) -> np.ndarray:
        n_samples, _ = x_train.shape
        centroid_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = x_train[centroid_idx]

        for _ in range(self.iter):
            # assign point to nearest centroids
            # euclidean distance computation
            dist = np.linalg.norm(x_train[:,np.newaxis] - self.centroids, axis=2)
            points = np.argmin(dist, axis=1)
            # compute mean of centroids
            temp_centroid = np.array([np.mean(x_train[np.equal(points, i)], axis= 0) for i in range(self.n_clusters)])
            self.centroids = temp_centroid
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(x_test[:,np.newaxis] - self.centroids, axis=2)
        return np.argmin(dist, axis=1)