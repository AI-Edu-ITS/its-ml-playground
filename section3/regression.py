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