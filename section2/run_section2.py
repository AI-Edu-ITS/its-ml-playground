import argparse
import numpy as np
import os
import sys

# enable import from outside folder
sys.path.insert(0, os.getcwd())

from agglomerative import AgglomerativeClustering, visualize_preds_agglo
from kmeans import KMeans, visualize_preds_kmeans, visualize_elbow_kmeans
from tools.utils import load_csv_data, train_test_split
from tools.clustering_metrics import evaluation_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 2 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (kmeans, agglo)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)
    
    # kmeans args
    # parser.add_argument('-nc', '--n_cluster', help='Define number of cluster for kmeans', default=2, type=int)
    parser.add_argument('-it', '--iter', help='Define number of iteration of training in Kmeans', default=300, type=int)

    # agglomerative args
    parser.add_argument('-li', '--linkage', help='Define Linkage type (complete, single)', default='complete', type=str)
    parser.add_argument('-dl', '--data_limit', help='Only in Agglomerative Clustering, Limit number of data used in training due to slow speed', default=200, type=int)

    # misc args
    parser.add_argument('-dm', '--dist_metric', help='Distance metric used (euclid, manhattan, minkowski)', type=str, default='euclid')
    parser.add_argument('-p', '--p_value', help='Determine p value for minkowski distance', default=3, type=int)
    
    args = parser.parse_args()

    # define dataset first (We use Shop Customer Dataset for Clustering Learning)
    dataset_path = './dataset/shop_customer_dataset.csv'
    x_columns = ['Annual Income ($)','Spending Score (1-100)','Work Experience','Family Size']
    y_columns = 'Gender'
    x_data, y_data = load_csv_data(dataset_path, x_columns, y_columns)
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)

    if args.mode == 'predict':
        if args.algo == 'kmeans':
            kmeans_pred = KMeans(len(np.bincount(y_data)), args.iter)
            centroid, inertia = kmeans_pred.fit(x_data)
            result = kmeans_pred.predict(x_test)
            print(result)
            evaluation_report(args.algo, x_test, result, y_test, args.dist_metric)
        elif args.algo == 'agglo':
            # we limit the data only for agglomerative clustering due its slow speed compared to kmeans
            x_data = x_data[:args.data_limit, :]
            agglo_pred = AgglomerativeClustering(len(np.bincount(y_data)), args.linkage, args.dist_metric)
            labels = agglo_pred.fit(x_data)
            result = agglo_pred.predict(x_data)
            evaluation_report(args.algo, x_data, result, y_data, args.dist_metric)
    elif args.mode == 'vis':
        if args.algo == 'kmeans':
            kmeans_pred = KMeans(args.n_cluster, args.iter)
            centroid, inertia = kmeans_pred.fit(x_data)
            result = kmeans_pred.predict(x_test)
            visualize_preds_kmeans(x_test, result, centroid)
            visualize_elbow_kmeans(x_data, args.n_cluster)
        elif args.algo == 'agglo':
            x_data = x_data[:args.data_limit, :]
            agglo_pred = AgglomerativeClustering(args.n_cluster, args.linkage, args.dist_metric)
            centroids = agglo_pred.fit(x_data)
            result = agglo_pred.predict(x_data)
            visualize_preds_agglo(x_data, result)