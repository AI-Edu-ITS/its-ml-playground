import numpy as np

def binary_cross_entropy_loss(preds: np.ndarray, y_test: np.ndarray, eps: float) -> float:
    '''
        For only non exclusive class or maximum 2 class only loss (for example class 0 and 1 or class True and False)
    '''
    y1 = y_test * np.log(preds + eps)
    y2 = (1 - y_test) * np.log(1 - preds + eps)
    return -np.mean(y1 + y2)

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