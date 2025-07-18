import argparse

from datasets import download_datasets, load_dataset, sort_dataset, simplify_whr_classes
# from section1.visualize import b

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Section 1 Argument Parser")

    # add command
    parser.add_argument('-m', '--mode', help='Choose mode for running program (download, show, sort, plot_2d, plot_3d)', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='Dataset Path for Processing', type=str)
    parser.add_argument('-ts', '--type_sort', help='Type of dataset sorting (asc = ascending, desc = descending)', type=str)
    parser.add_argument('-c', '--column', help='Fill column you want to processed based on dataset', type=str)
    args = parser.parse_args()

    # execute command
    if args.mode == 'download':
        download_datasets()
        simplify_whr_classes('./dataset/temp_dataset_whr.csv')
    elif args.mode == 'show':
        load_dataset(args.dataset)
    elif args.mode == 'sort':
        sort_dataset(args.dataset, args.type_sort, args.column)