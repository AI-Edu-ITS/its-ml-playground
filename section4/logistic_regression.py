import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Enable import from other directory
sys.path.insert(0, os.getcwd())

from tools.loss import log_loss
from tools.activations import choose_activation

class LogisticRegression():
    '''
        Class for Logistic Regression algorithm. This class has similarity with supervised regression such as Simple Linear Regression or Multi Linear Regression.
        However, Logistic Regression purpose is to calculate the probability of an event like classification. For the equation itself similar to
        Linear Regression, but we add sigmod activation to this algorithm and produce equation as follow:

        y=σ(β1x1+...+βnxn + b)
    '''
    def __init__(self, learning_rate: float = 0.01, iteration: int = 100, threshold: float = 0.5, epsilon: float = 1e-8, activation: str = 'sigmoid', verbose: bool = True):
        self.lr = learning_rate
        self.epochs = iteration
        self.threshold = threshold
        self.epsilon = epsilon
        self.activation = activation
        self.verbose = verbose
        self.weight = np.ndarray
        self.bias = None
        self.loss_list = []

    def forward_pass(self, x_data: np.ndarray) -> np.ndarray:
        temp_layer: np.ndarray = np.matmul(x_data, self.weight) + self.bias
        out_layer = choose_activation(temp_layer, self.activation, 'forward')
        return out_layer
    
    def backward_pass(self, x_data: np.ndarray, y_data: np.ndarray, result: np.ndarray):
        n_samples, _ = x_data.shape
        # grad computation
        d_weight = np.matmul(x_data.T, np.subtract(result, y_data)) / n_samples
        d_bias = np.sum(np.subtract(result, y_data)) / n_samples
        # update weight and bias
        self.weight -= self.lr * d_weight
        self.bias -= self.lr * d_bias
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        _, n_features = x_train.shape
        self.weight = np.zeros(n_features, dtype=np.float128)
        self.bias = 0

        # gradient descent
        for epoch in range(self.epochs):
            # do forward pass
            out_result = self.forward_pass(x_train)
            # calc loss
            loss = log_loss(out_result, y_train, self.epsilon)
            self.loss_list.append(loss)
            # backward pass
            self.backward_pass(x_train, y_train, out_result)
            if self.verbose == True:
                print(f'Loss in Epoch {epoch} = {self.loss_list[epoch]}')

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        preds = self.forward_pass(x_test)
        return np.round(preds).astype(int)
    
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        preds_proba = self.forward_pass(x_test)
        return preds_proba

def visualize_activation_logistic(data: np.ndarray):
    plt.title('Various Activation in Logistic Regression')
    plt.plot(data, choose_activation(data, 'sigmoid', 'forward'), label='Sigmoid', color='red')
    plt.plot(data, choose_activation(data, 'relu', 'forward'), label='ReLu', color='blue')
    plt.plot(data, choose_activation(data, 'tanh', 'forward'), label='Tanh', color='green')
    plt.plot(data, choose_activation(data, 'softmax', 'forward'), label='Softmax', color='orange')
    plt.legend()
    plt.xlabel('X Val')
    plt.ylabel('Y Val')
    plt.show()