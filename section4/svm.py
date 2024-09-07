import numpy as np

class SVM():
    '''
        Class for Support Vector Machine. This implementation only can be used in prediction 2 type of class (yes, no) or (0,1).
        In addition, this implementation is svm without kernels
    '''
    def __init__(self, iterations: int = 100, learning_rate: float = 0.01, random_seed: int = 42, C: float = 0.01):
        self.iter = iterations
        self.lr = learning_rate
        self.random_seed = random_seed
        self.regularization = C # tradeoff
        self.weight = np.ndarray
        self.bias = float

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        _, n_features = x_train.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            # gradient descent
            for idx, x in enumerate(x_train):
                if y_train[idx] * (np.dot(x, self.weight) - self.bias) >= 1:
                    d_weight = 2 * self.regularization * self.weight
                    d_bias = 0
                    self.weight -= self.lr * d_weight
                else:
                    d_weight = 2 * self.regularization * self.weight - np.dot(x, y_train[idx])
                    d_bias = y_train[idx]
                    self.weight -= self.lr * d_weight
                    self.bias -= self.lr * d_bias

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # get the outputs
        output = np.dot(x_test, self.weight) - self.bias
        preds = [1 if val > 0 else -1 for val in output]
        return np.array(preds)