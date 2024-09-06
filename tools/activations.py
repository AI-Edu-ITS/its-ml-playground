import numpy as np

def choose_activation(x_data: np.ndarray, type: str = 'sigmoid') -> np.ndarray:
    if type == 'sigmoid': # sigmoid activation
        preds = sigmoid_activation(x_data)
    elif type == 'softmax':
        preds = softmax_activation(x_data)
    return preds

def sigmoid_activation(x_data: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x_data))

def softmax_activation(x_data: np.ndarray) -> np.ndarray:
    temp_exp = np.exp(x_data)
    return temp_exp / np.sum(temp_exp, axis=1).reshape(-1,1)