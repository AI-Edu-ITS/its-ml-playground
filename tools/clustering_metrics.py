import numpy as np
import os
import pandas as pd
import sys

from scipy.special import comb

# enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.utils import calc_distance

def calc_calinski_index(x_data: np.ndarray, result_class: np.ndarray) -> float:
    '''
        Function for calculate calinski-harabasz index also known as variance ratio criterion. we calculate this index
        with equation as follow:
        
        ch = (BCSS/WCSS) * ((N-K) / (K-1))

        where BCSS = between-cluster sum of squares; WCSS = within-cluster sum of squares; N = total number of observations; and K = total number of clusters.
        BCSS measures how well the clusters are separated from each other (the higher the better), while WCSS measures the compactness or cohesiveness of the clusters (the smaller the better).
    '''
    n_clusters = len(np.bincount(result_class))
    n_samples, n_features = x_data.shape
    x_mean: float = np.mean(x_data, axis=0)

    #define centroid
    centroid = np.zeros((n_clusters, n_features))
    bcss = 0
    wcss = 0
    for n_cluster in range(n_clusters):
        idx = np.where(np.equal(result_class, n_cluster))[0]
        centroid[n_cluster, :] = np.mean(x_data[idx], axis=0)
        bcss += len(idx) * np.power(np.linalg.norm(centroid[n_cluster] - x_mean),2)
        for x in x_data[idx, :]:
            wcss += np.power(np.linalg.norm(x - centroid[n_cluster]),2)
    calinski_harabasz_res = (bcss / (n_clusters - 1)) / (wcss / (n_samples - n_clusters)) 
    return round(calinski_harabasz_res, 3)

def calc_davies_index(x_data: np.ndarray, result_class: np.ndarray, dist_metric: str = 'euclid') -> float:
    '''
        Function for calculate davies-bouldin index which calculated as the average similarity of each cluster with a cluster most similar to it.
    '''
    final_result = []
    n_clusters = len(np.bincount(result_class))
    n_cluster = [x_data[result_class == n] for n in range(n_clusters)]
    centroids = [np.mean(n, axis=0) for n in n_cluster]
    variances = [np.mean([calc_distance(p, centroids[i], dist_metric) for p in k]) for i, k in enumerate(n_cluster)]

    for i in range(n_clusters):
        temp_result = []
        for j in range(n_clusters):
            if j != i:
                temp_result.append((variances[i] + variances[j]) / calc_distance(centroids[i], centroids[j], dist_metric))
        final_result.append(max(temp_result))
    davies_bouldin_res = np.mean(final_result)
    return round(davies_bouldin_res, 3)

def calc_silhouette_score(x_data: np.ndarray, result_class: np.ndarray, dist_metric: str = 'euclid') -> float:
    '''
        Function for calculate silhouette score in clustering.
    '''
    mean_similarity = []
    mean_not_similarity = []
    n_clusters = len(np.bincount(result_class))
    # save mean similarity data between n with all data points in same cluster
    for n_sim in range(n_clusters):
        temp_sim_data = x_data[np.equal(result_class, n_sim)]
        for l1_sim in range(len(temp_sim_data)):
            temp_sim_sum = 0
            temp_sim_row = temp_sim_data[l1_sim]
            for l2_sim in range(len(temp_sim_data)):
                temp_sim_sum += calc_distance(temp_sim_row, temp_sim_data[l2_sim], dist_metric)
            temp_sim_sum /= (len(temp_sim_data) - 1)
            mean_similarity.append(temp_sim_sum)
    # save mean dissimalirity data between n to some cluster C
    for n_diss in range(n_clusters):
        temp_diss_data = x_data[np.equal(result_class, n_diss)]
        for l1_diss in range(len(temp_diss_data)):
            temp_diss_row = temp_diss_data[l1_diss]
            min_diss_distance = []
            for n_diss_other in range(n_clusters):
                if n_diss != n_diss_other:
                    temp_diss_data_other = x_data[np.equal(result_class, n_diss_other)]
                    temp_diss_sum = 0
                    for l2_diss in range(len(temp_diss_data_other)):
                        temp_diss_sum += calc_distance(temp_diss_row, temp_diss_data_other[l2_diss])
                    temp_diss_sum /= (len(temp_diss_data_other))
                    min_diss_distance.append(temp_diss_sum)
            mean_not_similarity.append(min(min_diss_distance))
    score = pd.DataFrame({'mean_sim':mean_similarity,'mean_disim':mean_not_similarity})
    score['silhouette_score'] = (score['mean_disim'] - score['mean_sim'])/ score.max(axis = 1)
    mean_score = score['silhouette_score'].mean()
    return round(mean_score,3)

def calc_rand_index_score(y_data: np.ndarray, result_class: np.ndarray) -> float:
    '''
        Function for calculate rand index score.
    '''
    tp_fp = comb(np.bincount(y_data), 2).sum()
    tp_fn = comb(np.bincount(result_class), 2).sum()
    A = np.c_[(y_data, result_class)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(y_data))
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    rand_score = (tp + tn) / (tp + fp + fn + tn)
    return round(rand_score, 3)

def evaluation_report(algo: str, x_data: np.ndarray, result_class: np.ndarray, y_data: np.ndarray, dist_metric: str = 'euclid'):
    if algo == 'kmeans':
        type_algo = 'K-Means Clustering'
    elif algo == 'agglo':
        type_algo = 'Agglomerative Clustering'
    print(f'Clustering Evaluation for {type_algo} Algorithm')
    print(f'Silhouette Score = {calc_silhouette_score(x_data, result_class, dist_metric)}')
    print(f'Calinski-Harabasz Index = {calc_calinski_index(x_data, result_class)}')
    print(f'Davies-Bouldin Index = {calc_davies_index(x_data, result_class, dist_metric)}')
    print(f'Rand Index = {calc_rand_index_score(y_data, result_class)}')