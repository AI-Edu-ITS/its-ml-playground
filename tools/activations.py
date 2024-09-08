import numpy as np

def choose_activation(x_data: np.ndarray, activation: str, state: str):
    result = np.ndarray
    if activation == 'sigmoid':
        if state == 'forward':
            result = sigmoid(x_data)
        else:
            result = sigmoid_derivative(x_data)
    elif activation == 'relu':
        if state == 'forward':
            result = relu(x_data)
        else:
            result = relu_derivative(x_data)
    elif activation == 'tanh':
        if state == 'forward':
            result = tanh(x_data)
        else:
            result = tanh_derivative(x_data)
    return result

# sigmoid (forward pass and its backward pass)
def sigmoid(x_data: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x_data))

def sigmoid_derivative(x_data: np.ndarray) -> np.ndarray: # gradient calc in sigmoid
    theta = sigmoid(x_data)
    return theta * (1 - theta)

# relu (forward pass and its backward pass)
def relu(x_data: np.ndarray) -> np.ndarray:
    temp = np.zeros(x_data.shape, dtype=np.float32)
    return np.maximum(temp, x_data)

def relu_derivative(x_data: np.ndarray) -> np.ndarray: # gradient calc in relu
    return np.greater(x_data, 0.).astype(np.float32)

# tanh (forward pass and its backward pass)
def tanh(x_data: np.ndarray) -> np.ndarray:
    return np.tanh(x_data)

def tanh_derivative(x_data: np.ndarray) -> np.ndarray:
    return 1. / (np.cosh(x_data) ** 2)