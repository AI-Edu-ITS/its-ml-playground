import numpy as np
from tools.classification_metrics import calc_accuracy

def binary_cross_entropy_loss(preds: np.ndarray, y_data: np.ndarray, eps: float = 1e-8) -> float:
    '''
        For only non exclusive class or maximum 2 class only loss (for example class 0 and 1 or class True and False)
    '''
    preds = preds.reshape(-1,1)
    y_data = y_data.reshape(-1,1)
    preds = np.clip(preds, eps, 1 - eps)
    y1 = (1 - y_data) * np.log(1 - preds + eps)
    y2 = y_data * np.log(preds + eps)
    return -np.mean(y1 + y2, axis=0)

def categorical_cross_entropy_loss(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        For multiclass loss if label treat as one-hot encoding
    '''
    return -1 * np.mean(y_test * np.log(preds))

def sparse_categorical_cross_entopy_loss(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        For multiclass loss if label treat as integer (number)
    '''
    # label to one-hot
    y_test_onehot = np.zeros_like(preds)
    y_test_onehot[np.arange(len(y_test)), y_test] = 1

    return -np.mean(np.sum(y_test_onehot * np.log(preds), axis=-1))

def calc_huber_loss(preds: np.ndarray, y_true: np.ndarray, delta: float = 1.0):
    '''
        Loss in regression which more robust to outlier compare to Mean Squared Error (MSE)
    '''
    error = np.abs(preds - y_true)
    quadratic = np.minimum(error, delta)
    return np.mean(0.5 * quadratic ** 2 + delta * (error - quadratic))

def log_loss(preds: np.ndarray, y_test: np.ndarray, eps: float) -> float:
    temp_preds = [max(i, eps) for i in preds]
    temp_preds = [min(i, 1 - eps) for i in temp_preds]
    temp_preds = np.array(temp_preds)
    return -np.mean(y_test * np.log(preds) + (1 - y_test) * np.log(1 - temp_preds))

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0
class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return calc_accuracy(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)