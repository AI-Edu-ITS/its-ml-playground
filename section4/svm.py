import matplotlib.pyplot as plt
import numpy as np

class SVMNoKernel():
    '''
        Class for Support Vector Machine, this implementation is svm without kernels and only support 2 class
    '''
    def __init__(self, iterations: int = 100, learning_rate: float = 0.01, C: float = 0.01):
        self.iter = iterations
        self.lr = learning_rate
        self.regularization = C
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
                else:
                    d_weight = 2 * self.regularization * self.weight - np.dot(x, y_train[idx])
                    d_bias = y_train[idx]
                self.weight -= self.lr * d_weight
                self.bias -= self.lr * d_bias

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        # get the outputs
        output = np.dot(x_test, self.weight) - self.bias
        label_signs = np.sign(output)
        preds = np.where(label_signs <= -1, 0, 1)
        return np.array(preds)

def visualize_svm_result(x_data: np.ndarray, y_data: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(x_data[:,0], x_data[:,1], marker='o', c=y_data)

    x0_1 = np.amin(x_data[:,0])
    x0_2 = np.amax(x_data[:,0])

    x1_1 = hyperplane(x0_1, weight, bias, 0)
    x1_2 = hyperplane(x0_2, weight, bias, 0)

    x1_1_m = hyperplane(x0_1, weight, bias, -1)
    x1_2_m = hyperplane(x0_2, weight, bias, -1)

    x1_1_p = hyperplane(x0_1, weight, bias, 1)
    x1_2_p = hyperplane(x0_2, weight, bias, 1)

    ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(x_data[:,1])
    x1_max = np.amax(x_data[:,1])
    ax.set_ylim([x1_min-3,x1_max+3])
    
    plt.title('Plot Linear SVM No Kernel')
    plt.show()

def hyperplane(x_data: np.ndarray, weight: np.ndarray, bias: np.ndarray, offset: int):
    return (-weight[0] * x_data + bias + offset) / weight[1]