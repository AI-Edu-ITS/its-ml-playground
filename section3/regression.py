import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression():
    '''
        Simple Linear Regression class. This algorithm has an equation as follow -> y = β0 + β1x

        y = dependent variable; β0 = intercept; βi = slope
    '''
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x_train_mean = x_train.mean()
        y_train_mean = y_train.mean()

        # calculate slope & intercept
        num = np.sum((x_train - x_train_mean) * (y_train - y_train_mean))
        denom = np.sum((x_train - x_train_mean) ** 2)
        self.slope = num / denom
        self.intercept = y_train_mean - self.slope * x_train_mean
    
    def predict(self, x_test: np.ndarray):
        return self.intercept + self.slope * x_test

class MultiLinearRegression():
    '''
        Multi Linear Regression class. This algorithm has an equation as follow -> y = β0 + β1xi1 + ... + βpxip

        y = dependent variable; β0 = intercept; βi = slope; xi = features variable

        Input: learning rate of model, error rate threshold
    '''
    def __init__(self, learning_rate: float = 0.00001, epsilon: float = 0.9):
        self.slope = None
        self.intercept = None
        self.lr = learning_rate
        self.eps = epsilon
        self.weights = np.ndarray
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        regression: np.ndarray = np.c_[x_train, np.ones(len(x_train))]
        self.weights = np.ones(regression.shape[1])
        # gradient descent
        loss = 1
        while(loss > self.eps):
            y_pred = np.matmul(regression, self.weights.T)
            partial = np.matmul(regression.T, (y_train - y_pred))
            loss = np.sum(np.sqrt(np.square(partial)))
            self.weights = self.weights.T + (self.lr * partial)

            if(np.isnan(loss)):
                print('Model diverged, try different LR')
    
    def predict(self, x_test: np.ndarray):
        return np.matmul(self.weights[:-1], (x_test.T + self.weights[-1]))

def visualize_simple_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    lr_preds = SimpleLinearRegression()
    lr_preds.fit(x_train, y_train)
    result = lr_preds.predict(x_test)

    # plot scatter
    plt.scatter(x_test, y_test, label='Value')
    plt.plot(x_test, result, color='red', marker='o', label='Simple Linear Regression Line')
    plt.xlabel('X-Test Data')
    plt.ylabel('Prediction Res')
    plt.show()

def visualize_multi_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    lr_preds = MultiLinearRegression()
    lr_preds.fit(x_train, y_train)
    result = lr_preds.predict(x_test)

    # plot scatter
    plt.scatter(y_test, result, label='Value')
    # plt.plot(x_test, result, color='red', marker='o', label='Multi Linear Regression Line')
    plt.xlabel('Test Data')
    plt.ylabel('Prediction Data')
    plt.show()