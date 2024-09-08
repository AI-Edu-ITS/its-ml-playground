import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.loss import log_loss, categorical_cross_entropy_loss

class LogisticRegression():
    '''
        Class for Logistic Regression algorithm. This class has similarity with supervised regression such as Simple Linear Regression or Multi Linear Regression.
        However, Logistic Regression purpose is to calculate the probability of an event like classification. For the equation itself similar to
        Linear Regression, but we add sigmod activation to this algorithm and produce equation as follow:

        y=σ(β1x1+...+βnxn + b)
    '''
    def __init__(self, learning_rate: float = 0.01, iteration: int = 100, threshold: float= 0.5, epsilon: float = 1e-9, verbose: bool = False):
        self.lr = learning_rate
        self.iter = iteration
        self.eps = epsilon
        self.threshold = threshold
        self.verbose = verbose
        self.weight = None
        self.bias = None
        self.loss_list = {}
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = x_train.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for i in range(self.iter):
            temp_class_pred = np.dot(x_train, self.weight) + self.bias
            # apply sigmoid activation
            y_pred = self.sigmoid_activation(temp_class_pred)
            d_weight = 1 / float(n_samples) * np.dot(x_train.T, (y_pred - y_train))
            d_bias = 1 / float(n_samples) * np.sum((y_pred - y_train))
            self.weight -= self.lr * d_weight
            self.bias -= self.lr * d_bias 

            # compute loss
            loss = log_loss(y_pred, y_train, self.eps)
            self.loss_list[i] = loss
            if self.verbose == True:
                print(f'Loss in iteration {i} = {loss}')

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        temp_y = np.dot(x_test, self.weight) + self.bias
        y_preds = self.sigmoid_activation(temp_y) >= self.threshold
        return np.array(y_preds)
    
    def sigmoid_activation(self, x_data: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-x_data)).reshape(-1,1)

class MultiLogisticRegression():
    '''
        Extended version of Logistic Regression which can hangle multi-class classification (more than 2 classes)
    '''
    def __init__(self, learning_rate: float = 0.01, iteration: int = 100, batch_size: int = 32, rand_seed: int = 4, verbose: bool = False):
        self.lr = learning_rate
        self.iter = iteration
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.weights = np.ndarray
        self.verbose = verbose
        self.loss_list = {}
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        np.random.seed(self.rand_seed)
        self.classes = np.unique(y_train)
        self.class_labels = {c: i for i,c in enumerate(self.classes)}
        # add bias and one-hot
        x_train = np.insert(x_train, 0, 1, axis=1)
        y_train = np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y_train).reshape(-1)]
        self.weights = np.zeros(shape=(len(self.classes), x_train.shape[1]))
        for i in range(1, self.iter + 1):
            # gradient descent for loss
            preds = self.calc_gradient_descent(x_train)
            loss = categorical_cross_entropy_loss(preds, y_train)
            if self.verbose == True:
                print(f'Iteration {i} loss = {loss}')
            self.loss_list[i] = loss
            
            idx = np.random.choice(x_train.shape[0], self.batch_size)
            x_batch, y_batch = x_train[idx], y_train[idx]
            # gradient descent for batch
            final_y_batch = self.calc_gradient_descent(x_batch)
            error = y_batch - final_y_batch
            update = (self.lr * np.dot(error.T, x_batch))
            self.weights += update

    def calc_gradient_descent(self, x_test: np.ndarray) -> np.ndarray:
        temp_preds = np.dot(x_test, self.weights.T).reshape(-1,len(self.classes))
        final_preds = self.calc_softmax(temp_preds)
        return final_preds
    
    def calc_softmax(self, x_data: np.ndarray) -> np.ndarray:
        temp_exp = np.exp(x_data)
        return temp_exp / np.sum(temp_exp, axis=1).reshape(-1,1)

    def predict(self, x_test: np.ndarray):
        x_test = np.insert(x_test, 0, 1, axis=1)
        probs = self.calc_gradient_descent(x_test)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(probs, axis=1))

def visualize_logits_loss(loss_list: dict):
    plt.title('Loss Value of Logistic Regression')
    x_data = np.array(list(loss_list.keys()))
    y_data = np.array(list(loss_list.values()))
    plt.plot(x_data, y_data)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.show()