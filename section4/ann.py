import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

# Enable import outside directory
sys.path.insert(0, os.getcwd())

from tools.activations import choose_activation

class MLPClassifier: 
    '''
        Implementation of Multi-Layer Perceptron for multi class clasiification. You can choose the activations section
        (whether using Sigmoid, ReLu, or Tanh). Important notes:
        Input layer must match number of x data features (check it by using x_data.shape[1] or simply pass it). Output
        layer must match number of class available in y data (simply check it by len(np.unique(y_data)) or simply pass it)
    '''
    def __init__(
            self, 
            input_layer: int = 3,
            hidden_layer: int = 3,
            output_layer: int = 5,
            learning_rate: float = 0.005,
            epochs: int = 300,
            bias_hidden_layer: int = -1,
            bias_ouput_layer: int = -1,
            activation: str = 'relu'
        ):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.lr = learning_rate
        self.epochs = epochs
        self.bias_hidden = bias_hidden_layer
        self.bias_output = bias_ouput_layer
        self.activation = activation
        self.error_list = []
        self.epoch_list = []

        # init weight
        self.hidden_weights = [[2  * random.random() - 1 for i in range(self.hidden_layer)] for j in range(self.input_layer)]
        self.output_weights = [[2  * random.random() - 1 for i in range(self.output_layer)] for j in range(self.hidden_layer)]
        self.hidden_bias = np.array([self.bias_hidden for i in range(self.hidden_layer)])
        self.output_bias = np.array([self.bias_output for i in range(self.output_layer)])
    
    # Define Backpropagation process algoritm
    def backpropagation(self, x_data: np.ndarray):
        delta_t = []
        
        # Stage 1 - Error: OutputLayer
        error_val = self.output - self.output_l2
        delta_t = -1 * error_val * choose_activation(self.output_l2, self.activation, 'backward')
        
        # Stage 2 - Update weights OutputLayer and HiddenLayer
        for i in range(self.hidden_layer):
            for j in range(self.output_layer):
                self.output_weights[i][j] -= self.lr * (delta_t[j] * self.output_l1[i])
                self.output_bias[j] -= (self.lr * delta_t[j])

        # Stage 3 - Error: HiddenLayer
        delta_hidden = np.matmul(self.output_weights, delta_t) * choose_activation(self.output_l1, self.activation, 'backward')

        # Stage 4 - Update weights HiddenLayer and InputLayer(x)
        for i in range(self.output_layer):
            for j in range(self.hidden_layer):
                self.hidden_weights[i][j] -= self.lr * (delta_hidden[j] * x_data[i])
                self.hidden_bias[j] -= (self.lr * delta_hidden[j])
    
    # Define predict function for prediction test data
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        my_predictions = []
        
        # Just doing Forward Propagation
        forward = np.matmul(x_test, self.hidden_weights) + self.hidden_bias
        forward = np.matmul(forward, self.output_weights) + self.output_bias
        
        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])
                
        return np.array(my_predictions)
    
    # turn labels into 0 1 array formats and out as same as index get
    def one_hot_encode_label(self, y_data: np.ndarray) -> dict:
        labels_dict = {}
        classes = np.unique(y_data)
        temp_data = np.eye(len(classes))[classes.reshape(-1)]
        for i in range(len(classes)):
            labels_dict[i] = temp_data[i]
        return labels_dict

    # training process with train data
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # class count
        total_error = 0
        hidden_weight_loss = []
        output_weight_loss = []
        # one hot label
        self.output = np.zeros(len(np.unique(y_train)))
        self.one_hot_labels = self.one_hot_encode_label(y_train)
        for epoch in range(1, self.epochs + 1):
            for idx, inputs in enumerate(x_train):
                # Forward Pass
                temp_l1 = np.dot(inputs, self.hidden_weights) + self.hidden_bias.T
                self.output_l1 = choose_activation(temp_l1, self.activation, 'forward')
                temp_l2 = np.dot(self.output_l1, self.output_weights) + self.output_bias.T
                self.output_l2 = choose_activation(temp_l2, self.activation, 'forward')
                
                # Stage 2 - One-Hot-Encoding
                self.output = self.one_hot_labels[y_train[idx]]
                
                square_error = 0
                for i in range(self.output_layer):
                    erro = (self.output[i] - self.output_l2[i]) ** 2
                    square_error += (0.05 * erro)
                    total_error += square_error
                # Backpropagation : Update Weights
                self.backpropagation(inputs)
                
            total_error = (total_error / len(x_train))
            
            # Print error value for each epoch
            print("Epoch ", epoch, "- Total Error: ",total_error)
            self.error_list.append(total_error)
            self.epoch_list.append(epoch)
                
            hidden_weight_loss.append(self.hidden_weights)
            output_weight_loss.append(self.output_weights)
            
        # self.show_err_graphic(error_array,epoch_array)
        
        # Print weight Hidden layer acquire during training
        print('')
        print('weight value of Hidden layer acquire during training: ')
        print(hidden_weight_loss[0])
        
        # Plot weight Output layer acquire during training
        print('')
        print('weight value of Output layer acquire during training: ')
        print(output_weight_loss[0])

def visualize_loss(epoch_list: list, error_list: list):
    plt.figure(figsize=(9,4))
    plt.plot(epoch_list, error_list, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Epochs vs Loss')
    plt.show()