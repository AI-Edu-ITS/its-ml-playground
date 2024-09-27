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
    elif activation == 'softmax':
        if state == 'forward':
            result = softmax(x_data)
        else:
            result = softmax_derivative(x_data)
    return result

# softmax (forward pass and its backward pass)
def softmax(x_data: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x_data - np.max(x_data))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_derivative(x_data: np.ndarray) -> np.ndarray:
    return softmax(x_data) * (1 - softmax(x_data))

# sigmoid (forward pass and its backward pass)
def sigmoid(x_data: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x_data * 1.0))

def sigmoid_derivative(x_data: np.ndarray) -> np.ndarray: # gradient calc in sigmoid
    return sigmoid(x_data) * (1 - sigmoid(x_data))

# relu (forward pass and its backward pass)
def relu(x_data: np.ndarray) -> np.ndarray:
    return np.maximum(0, x_data)

def relu_derivative(x_data: np.ndarray) -> np.ndarray: # gradient calc in relu
    return 1. * (x_data > 0)

# tanh (forward pass and its backward pass)
def tanh(x_data: np.ndarray) -> np.ndarray:
    return np.tanh(x_data)

def tanh_derivative(x_data: np.ndarray) -> np.ndarray:
    return 1 - np.power(tanh(x_data), 2)