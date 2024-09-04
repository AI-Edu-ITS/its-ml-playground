import numpy as np
import os
import sys

# enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.utils import calc_distance

def calc_calinski_index(x_data: np.ndarray, y_data: np.ndarray) -> float:
    '''
        Function for calculate calinski-harabasz index also known as variance ratio criterion. we calculate this index
        with equation as follow:
        
        ch = (BCSS/WCSS) * ((N-K) / (K-1))

        where BCSS = between-cluster sum of squares; WCSS = within-cluster sum of squares; N = total number of observations; and K = total number of clusters.
        BCSS measures how well the clusters are separated from each other (the higher the better), while WCSS measures the compactness or cohesiveness of the clusters (the smaller the better).
    '''
    n_clusters =len(np.unique(y_data))
    n_samples, n_features = x_data.shape
    x_mean = np.mean(x_data, axis=0)

    #define centroid
    centroid = np.zeros((n_clusters, n_features))
    bcss = 0
    wcss = 0
    for n_cluster in range(n_clusters):
        i = np.where(np.equal(y_data, n_cluster))[0]
        ni = len(i)
        centroid[n_cluster, :] = np.mean(x_data[i], axis=0)
        bcss += ni * np.linalg.norm(centroid[n_cluster] - x_mean)**2
        for x in x_data[i, :]:
            wcss += np.linalg.norm(centroid[n_cluster] - x)**2
    calinski_harabasz_res = bcss / wcss * (n_samples - n_clusters) / (n_clusters - 1)
    return round(calinski_harabasz_res, 3)

def calc_davies_index(x_data: np.ndarray, y_data: np.ndarray) -> float:
    '''
        Function for calculate davies-bouldin index which calculated as the average similarity of each cluster with a cluster most similar to it.
    '''
    final_result = []
    n_clusters = len(np.bincount(y_data))
    n_cluster = [x_data[y_data == n] for n in range(n_clusters)]
    centroids = [np.mean(n, axis=0) for n in n_cluster]
    variances = [np.mean([calc_distance(p, centroids[i]) for p in k]) for i, k in enumerate(n_cluster)]

    for i in range(n_clusters):
        temp_result = []
        for j in range(n_clusters):
            if j != i:
                temp_result.append((variances[i] + variances[j]) / calc_distance(centroids[i], centroids[j]))
        final_result.append(max(temp_result))
    davies_bouldin_res = np.mean(final_result)
    return round(davies_bouldin_res, 3)
