import matplotlib.pyplot as plt
import numpy as np

class KMeans():
    '''
        Class for running LLoyd Algorithm or known as K-Means Clustering. we used parameters inputs such as
        number of clusters, maximum iteration, and tolerance value for decalring model divergence. we also add distance
        metric to choose for calculate distance between centroid and p value for minkowski distance
    '''
    def __init__(self, n_clusters: int = 8, iterations: int = 100, dist_metric: str = 'euclid', p: int = 3) -> None:
        self.n_clusters = n_clusters
        self.iter = iterations
        self.dist_metric = dist_metric
        self.p = p
    
    def fit(self, x_train: np.ndarray) -> np.ndarray:
        n_samples, _ = x_train.shape
        centroid_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = x_train[centroid_idx]
        self.points = None
        self.inertia = np.inf

        for _ in range(self.iter):
            # assign point to nearest centroids
            # distance computation
            dist = np.linalg.norm(x_train[:,np.newaxis] - self.centroids, axis=2)
            self.points = np.argmin(dist, axis=1)
            temp_inertia = np.sum(np.min(dist, axis=1))
            # compute mean of centroids
            temp_centroid = np.zeros((self.n_clusters, x_train.shape[1]))
            for idx in range(self.n_clusters):
                centroid_mean = np.mean(x_train[np.equal(self.points, idx)], axis=0)
                temp_centroid[idx] = centroid_mean
            self.centroids = temp_centroid
            self.inertia = temp_inertia
        return self.centroids, self.inertia      
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(x_test[:,np.newaxis] - self.centroids, axis=2)
        return np.argmin(dist, axis=1)

def visualize_preds_kmeans(x_test: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
    classes = np.unique(labels)
    for idx in classes:
        plt.scatter(x_test[labels == idx, 0], x_test[labels == idx, 1], s=80, label=f'Cluster {idx}')
    plt.scatter(centroids[:,0], centroids[:,1], s=100, c='k')
    plt.title('Visualize Cluster from K-Means')
    plt.legend()
    plt.show()