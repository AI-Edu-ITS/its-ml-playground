import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.classification_metrics import calc_accuracy

class DecisionTreeClassifier():
    '''
        Class for Decision Tree Classifier algorithm. Using node approach to create tree. there are 3 type of criterion which can be chosen namely entropy, gini, and missclassification error.
        entropy criterion or called log loss criterion purpose is to measure impurity in decision tree, while gini critertion purpose is to minimize the probability of miscclasification.

        We calculate entropy criterion or log loss criterion as follow -> entropy = sum((-class_probability) * (log2(class_probability)))

        For calculate gini criterion as follow -> gini = 1 - sum(class_probability * class_probability))
    '''
    def __init__(self, criterion: str = 'gini', max_depth: int = 5, min_samples_split: int = 20):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.node_root = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.classes = np.unique(y_train)
        # reshape y_train first
        y_train = y_train.reshape(-1,1)
        concat_dataset = np.concatenate((x_train, y_train), axis=1)
        self.node_root = self.construct_node_tree(concat_dataset, 0)

    def construct_node_tree(self, dataset: np.ndarray, node_depth: int):
        x_data = dataset[:,:-1]
        y_data = dataset[:,-1]
        _, cols = x_data.shape

        # check is depth equal or less than maximum depth defined before
        if node_depth <= self.max_depth:
            temp_data = self.split_left_right_data(cols, dataset)
            if temp_data['gain'] > 0:
                left_node_tree = self.construct_node_tree(temp_data['left_data'], node_depth + 1)
                right_node_tree = self.construct_node_tree(temp_data['right_data'], node_depth + 1)
                return Node(
                    col_index=temp_data['col_idx'], 
                    threshold=temp_data['threshold'], 
                    left_node=left_node_tree,
                    right_node=right_node_tree,
                    gain=temp_data['gain']
                )
        # leaf node
        leaf_y_data = list(y_data)
        return Node(value_leaf_node=max(leaf_y_data, key=leaf_y_data.count))
    
    def split_left_right_data(self, cols: int, dataset: np.ndarray) -> dict:
        temp_data = {}
        # init temp data key first
        temp_data['col_idx'] = -1
        temp_data['gain'] = -1
        temp_data['left_data'] = -1
        temp_data['right_data'] = -1
        temp_data['threshold'] = -1
        maximum_gain = -float('inf')
        for col in range(cols):
            temp_feats = dataset[:, col]
            unique_data = np.unique(temp_feats)

            for threshold in unique_data:
                left_data, right_data = self.split_data(dataset, col, threshold)
                if len(left_data) > 0 and len(right_data) > 0:
                    parent_class, left_class, right_class = dataset[:,-1], left_data[:, -1], right_data[:,-1]
                    cur_gain = self.choose_criterion(parent_class, left_class, right_class, self.criterion)
                    if cur_gain > maximum_gain:
                        temp_data['col_idx'] = col
                        temp_data['gain'] = cur_gain
                        temp_data['threshold'] = threshold
                        temp_data['left_data'] = left_data
                        temp_data['right_data'] = right_data
                        maximum_gain = cur_gain
        return temp_data

    def split_data(self, dataset: np.ndarray, feat_idx: int, threshold: float):
        left_data = np.array([row for row in dataset if row[feat_idx] <= threshold])
        right_data = np.array([row for row in dataset if row[feat_idx] > threshold])
        return left_data, right_data

    # for choose criterion in decision tree, you can choose between entropy or gini. We will get shannon information gain
    def choose_criterion(self, parent_data, left_data, right_data, criterion_type: str = 'gini'):
        size_left = len(left_data) / len(parent_data)
        size_right = len(right_data) / len(parent_data)
        if criterion_type == 'gini':
            gain = self.gini_criterion(parent_data) - (size_left * self.gini_criterion(left_data) + size_right * self.gini_criterion(right_data))
        elif criterion_type == 'entropy':
            gain = self.entropy_criterion(parent_data) - (size_left * self.entropy_criterion(left_data) + size_right * self.entropy_criterion(right_data))
        return gain

    def gini_criterion(self, y_data: np.ndarray) -> float:
        classes = np.unique(y_data)
        num_y_data = len(y_data)
        temp_label = {}
        final_value = 0.0
        for y in y_data:
            if y in temp_label:
                temp_label[y] += 1
            else:
                temp_label[y] = 1
        for n in classes:
            proba = temp_label[n] / num_y_data
            final_value += proba * proba
        return 1 - final_value

    def entropy_criterion(self, y_data: np.ndarray):
        classes = np.unique(y_data)
        num_y_data = len(y_data)
        temp_label = {}
        final_value = 0.0
        for y in y_data:
            if y in temp_label:
                temp_label[y] += 1
            else:
                temp_label[y] = 1
        for n in classes:
            proba = temp_label[n] / num_y_data
            final_value += (-proba) * (np.log2(proba))
        return final_value
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(x_data)))
        for x_idx, x_data in enumerate(x_data):
            preds[x_idx] = self.predict_util(x_data, self.node_root)
        return preds

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        preds_proba = np.zeros((len(x_data), len(self.classes)))
        for x_idx, x_data in enumerate(x_data):
            res = self.predict_util(x_data, self.node_root)
            for class_idx, class_label in enumerate(self.classes):
                preds_proba[x_idx, class_idx] = np.mean(np.equal(res, class_label))
        return preds_proba

    def predict_util(self, data, node):
        if node.value_leaf_node != None:
            return node.value_leaf_node

        if data[node.col_index] <= node.threshold:
            return self.predict_util(data, node.left_node)
        else:
            return self.predict_util(data, node.right_node)

class Node():
    '''
        Class for create tree using node pricipal
    '''
    def __init__(self, col_index = None, threshold = None, left_node = None, right_node = None, gain = None, value_leaf_node = None):
        self.col_index = col_index
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.gain = gain
        self.value_leaf_node = value_leaf_node

def visualize_gini_vs_entropy_tree(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, max_depth: int):
    acc_gini_list = []
    acc_entropy_list = []
    idx_rec_list = []
    for idx in range(1, max_depth + 1):
        # calc gini first
        gini_preds = DecisionTreeClassifier('gini', idx)
        gini_preds.fit(x_train, y_train)
        result_gini = gini_preds.predict(x_test)
        acc_gini_list.append(calc_accuracy(result_gini, y_test))
        # calc entropy then
        entropy_preds = DecisionTreeClassifier('entropy', idx)
        entropy_preds.fit(x_train, y_train)
        result_entropy = entropy_preds.predict(x_test)
        acc_entropy_list.append(calc_accuracy(result_entropy, y_test))
        idx_rec_list.append(idx)
    plt.plot(idx_rec_list, acc_gini_list, label='Gini')
    plt.plot(idx_rec_list, acc_entropy_list, label='Entropy')
    plt.xlabel('Number of Maximum Tree Depth')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy vs Max Depth in Different Criterion')
    plt.legend()
    plt.show()