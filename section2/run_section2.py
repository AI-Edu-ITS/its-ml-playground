import argparse
import os
import sys

# enable import from outside folder
sys.path.insert(0, os.getcwd())

from agglomerative import AgglomerativeClustering
from kmeans import KMeans
from tools.utils import load_csv_data, train_test_split
from tools.clustering_metrics import evaluation_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Section 2 Argument Parser')
    parser.add_argument('-m', '--mode', help='Choose mode for running (predict, vis, compare)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Define Dataset Path', required=True, type=str)
    parser.add_argument('-a', '--algo', help='Run specific algo (kmeans, agglo)', type=str)
    parser.add_argument('-ts', '--train_split', help='Define training percentage over testing percentage', default=0.7, type=float)
    
    # kmeans args
    parser.add_argument('-nc', '--n_cluster', help='Define number of cluster for kmeans', default=8, type=int)
    parser.add_argument('-it', '--iter', help='Define number of iteration of training in Kmeans', default=100, type=int)

    # misc args
    parser.add_argument('-dm', '--dist_metric', help='Distance metric used (euclid, manhattan, minkowski)', type=str, default='euclid')
    parser.add_argument('-p', '--p_value', help='Determine p value for minkowski distance', default=3, type=int)
    
    args = parser.parse_args()

    # define dataset first (We use Shop Customer Dataset for Clustering Learning)
    x_columns = ['Annual Income ($)','Spending Score (1-100)','Work Experience','Family Size']
    y_columns = 'Gender'
    x_data, y_data = load_csv_data(args.dataset, x_columns, y_columns)
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, args.train_split)

    if args.mode == 'predict':
        if args.algo == 'kmeans':
            kmeans_pred = KMeans(args.n_cluster, args.iter, args.dist_metric)
            centroid, inertia = kmeans_pred.fit(x_data)
            result = kmeans_pred.predict(x_test)
            evaluation_report(x_test, result, args.dist_metric)
        elif args.algo == 'agglo':
            agglo_pred = AgglomerativeClustering(args.n_cluster)
            result = agglo_pred.fit_predict(x_data)
            evaluation_report(x_data, result, args.dist_metric)
