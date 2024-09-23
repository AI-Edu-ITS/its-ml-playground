import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances

# enable import from other directory
sys.path.insert(0, os.getcwd())
from tools.utils import calc_distance, calc_linkage

class AgglomerativeClustering:
    '''
        Class of Hierarchical Clustering using Agglomerative Clustering or bottom-up approach.
    '''
    def __init__(self, n_clusters: int = 4, linkage_type: str = 'complete', dist_metric: str = 'euclid', p: int = 3):
        self.n_clusters = n_clusters
        self.linkage_type = linkage_type
        self.dist_metric = dist_metric
        self.p = p
    
    def argmin(self, cluster_distance: np.ndarray):
        min_x, min_y = (0,0)
        min_val = 1e6
        for i in range(cluster_distance.shape[0]):
            for j in range(cluster_distance.shape[0]):
                if j != i:
                    if cluster_distance[i, j] < min_val:
                        min_val = cluster_distance[i, j]
                        min_x = i
                        min_y = j
        return min_val, min_x, min_y

    def calc_cluster_distance(self, member_clusters: dict, x_data: np.ndarray) -> np.ndarray:
        num_cluster = len(member_clusters)
        key_cluster = list(member_clusters.keys())
        final_distance = np.zeros((num_cluster, num_cluster))
        for i in range(num_cluster):
            # pick ith cluster
            i_element = member_clusters[key_cluster[i]]
            # compare to other cluster
            for j in range(num_cluster):
                j_element = member_clusters[key_cluster[j]]
                dist = euclidean_distances(x_data[i_element], x_data[j_element])
                dist_linkage = calc_linkage(self.linkage_type, dist)
                final_distance[i,j] = dist_linkage
        return final_distance
    
    def fit(self, x_data: np.ndarray) -> np.ndarray:
        n_samples, _ = x_data.shape
        member_clusters = {i: [i,] for i in range(n_samples)}
        temp_labels = np.zeros((n_samples - 1, 4))

        for i in range(n_samples - 1):
            key_cluster = list(member_clusters.keys())
            cluster_distance = self.calc_cluster_distance(member_clusters, x_data)
            _, temp_min_x, temp_min_y = self.argmin(cluster_distance)
            temp_x = key_cluster[temp_min_x]
            temp_y = key_cluster[temp_min_y]
            # update labels
            temp_labels[i, 0] = temp_x
            temp_labels[i, 1] = temp_y
            temp_labels[i, 2] = cluster_distance[temp_min_x, temp_min_y]
            temp_labels[i, 3] = len(member_clusters[temp_x]) + len(member_clusters[temp_y])
            # merge cluster
            member_clusters[i + n_samples] = member_clusters[temp_x] + member_clusters[temp_y]
            member_clusters.pop(temp_x)
            member_clusters.pop(temp_y)
        self.centroids = temp_labels
        return self.centroids

    def predict(self, x_data: np.ndarray):
        n_samples, _ = x_data.shape
        preds = np.zeros((n_samples))
        cluster_members = dict([(i,[i]) for i in range(n_samples)])
        for i in range(n_samples - self.n_clusters):
            x = int(self.centroids[i,0])
            y = int(self.centroids[i,1])
            cluster_members[n_samples + i] = cluster_members[x] + cluster_members[y]
            del cluster_members[x]
            del cluster_members[y]
        keys = list(cluster_members.keys())
        for i in range(len(keys)):
            samples_in_cluster = cluster_members[keys[i]]
            preds[samples_in_cluster] = i
        return np.array(preds, dtype=np.int64)

def visualize_preds_agglo(x_test: np.ndarray, labels: np.ndarray):
    classes = np.unique(labels)
    for idx in classes:
        plt.scatter(x_test[labels == idx, 0], x_test[labels == idx, 1], s=80, label=f'Cluster {idx}')
    plt.title('Visualize Cluster from Agglomerative Clustering')
    plt.legend()
    plt.show()