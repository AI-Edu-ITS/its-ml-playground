import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Enable import outside directory
sys.path.insert(0, os.getcwd())

from tools.activations import choose_activation
from tools.loss import SquareLoss

class MLPClassifier: 
    '''
        Implementation of Multi-Layer Perceptron for multi class clasiification using Stocastic Gradient Descent solver. You can choose the activations section
        (whether using Sigmoid, ReLu, or Tanh). Important notes:
        Input layer must match number of x data features. Output layer must match number of class available in y data
    '''
    def __init__(
            self, 
            input_layer: int = 3,
            hidden_layer: int = 3,
            output_layer: int = 5,
            learning_rate: float = 0.005,
            epochs: int = 300,
            activation: str = 'sigmoid',
            verbose: bool = False,
        ):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.verbose = verbose
        self.error_list = []
        self.epoch_list = []
        self.loss = SquareLoss()

        # init weight
        self.hidden_weights = np.random.randn(self.input_layer, self.hidden_layer)
        self.output_weights = np.random.randn(self.hidden_layer, self.output_layer)
        self.hidden_bias = np.zeros((self.hidden_layer))
        self.output_bias = np.zeros((self.output_layer))
    
    # Define Backpropagation process algoritm
    def backpropagation(self, x_data: np.ndarray):
        delta_out = []
        
        # Error: OutputLayer
        error_val = self.output - self.output_l2
        delta_out = -1 * error_val * choose_activation(self.output_l2, self.activation, 'backward')
        
        # Update weights OutputLayer and HiddenLayer
        for i in range(self.hidden_layer):
            for j in range(self.output_layer):
                self.output_weights[i][j] -= self.lr * (delta_out[j] * self.output_l1[i])
                self.output_bias[j] -= (self.lr * delta_out[j])

        # Error: HiddenLayer
        delta_hidden = np.matmul(self.output_weights, delta_out) * choose_activation(self.output_l1, self.activation, 'backward')

        # Update weights HiddenLayer and InputLayer(x)
        for i in range(self.output_layer):
            for j in range(self.hidden_layer):
                self.hidden_weights[i][j] -= self.lr * (delta_hidden[j] * x_data[i])
                self.hidden_bias[j] -= (self.lr * delta_hidden[j])
    
    # Define predict function for prediction test data
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        my_predictions = []
        
        # Just doing Forward Propagation
        layer_1 = np.dot(x_test, self.hidden_weights) + self.hidden_bias
        out_1 = choose_activation(layer_1, self.activation, 'forward')
        layer_2 = np.dot(out_1, self.output_weights) + self.output_bias
        out_2 = choose_activation(layer_2, self.activation, 'forward')
        
        for i in out_2:
            my_predictions.append(max(enumerate(i), key=lambda x:x[1])[0])
        print(my_predictions)
                
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
        prev_total_error = 0
        cur_total_error = 0
        counter_stop = 0
        self.hidden_weight_loss = []
        self.output_weight_loss = []
        # one hot label
        self.output = np.zeros(len(np.unique(y_train)))
        self.one_hot_labels = self.one_hot_encode_label(y_train)
        for epoch in range(1, self.epochs + 1):
            prev_total_error = cur_total_error
            for idx, inputs in enumerate(x_train):
                # Forward Pass
                temp_l1 = np.dot(inputs, self.hidden_weights) + self.hidden_bias.T
                self.output_l1 = choose_activation(temp_l1, self.activation, 'forward')
                temp_l2 = np.dot(self.output_l1, self.output_weights) + self.output_bias.T
                self.output_l2 = choose_activation(temp_l2, self.activation, 'forward')
                
                # One-Hot-Encoding
                self.output = self.one_hot_labels[y_train[idx]]
                
                for i in range(self.output_layer):
                    temp_loss = self.loss.gradient(self.output[i], self.output_l2[i])
                    # square_error += (0.05 * erro)
                    cur_total_error += temp_loss
                # Backpropagation : Update Weights
                self.backpropagation(inputs)
                
            cur_total_error = (cur_total_error / len(x_train))
            if cur_total_error == prev_total_error:
                counter_stop += 1
                if counter_stop == 3:
                    break
            
            # Print error value for each epoch
            if self.verbose == True:
                print("Epoch ", epoch, "- Total Error: ",cur_total_error)
            self.error_list.append(cur_total_error)
            self.epoch_list.append(epoch)
                
            self.hidden_weight_loss.append(self.hidden_weights)
            self.output_weight_loss.append(self.output_weights)
        
        # Print weight Hidden layer acquire during training
        if self.verbose == True:
            print('')
            print('weight value of Hidden layer acquire during training: ')
            print(self.hidden_weight_loss[0])
            
            # Plot weight Output layer acquire during training
            print('')
            print('weight value of Output layer acquire during training: ')
            print(self.output_weight_loss[0])

def visualize_loss(epoch_list: list, error_list: list):
    plt.plot(epoch_list, error_list, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Epochs vs Loss')
    plt.show()