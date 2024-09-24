import numpy as np
import os
import sys

# enable import from another directory
sys.path.insert(0, os.getcwd())

from tools.classification_metrics import calc_accuracy, calc_f1score

class SelfTrainingClassifier():
    '''
        Self training classifier class. Use any supervised algorithm to proceed (only accept supervised algorithm
        which implement fit and predict_proba function)
    '''
    def __init__(self, classifier, iteration: int = 10, verbose: bool = False):
        self.classifier = classifier
        self.train_acc_list = []
        self.f1_score_list = []
        self.pseudo_labels = []
        self.epochs = iteration
        self.verbose = verbose

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_unlabeled: np.ndarray):
        temp_unlabeled = x_unlabeled
        temp_x_data = x_train
        temp_y_data = y_train
        high_proba_idx = [] # for temporary save index from unlabeled data
        for epoch in range(self.epochs):
            # train using chosen classifier
            self.classifier.fit(temp_x_data, temp_y_data)
            preds_train = self.classifier.predict(temp_x_data)
            self.train_acc_list.append(calc_accuracy(preds_train, temp_y_data))
            self.f1_score_list.append(calc_f1score(preds_train, temp_y_data))

            # predict unlabeled data
            pred_proba_unlabeled: np.ndarray = self.classifier.predict_proba(temp_unlabeled)
            preds_unlabeled: np.ndarray = self.classifier.predict(temp_unlabeled)
            for idx in range(len(pred_proba_unlabeled)):
                if(np.any(pred_proba_unlabeled[idx] > 0.99) == True):
                    self.pseudo_labels.append(preds_unlabeled[idx])
                    high_proba_idx.append(idx)
                    temp_x_data = np.vstack([temp_x_data, x_unlabeled[idx]])
                    temp_y_data = np.append(temp_y_data, preds_unlabeled[idx])
                else:
                    continue
            temp_unlabeled = np.delete(temp_unlabeled, (high_proba_idx), axis=0)
            high_proba_idx.clear()
            if self.verbose == True:
                print(f'Epoch - {epoch}')
                print(f'Training Data = {temp_x_data.shape[0]} data')
                print(f'Remaining Unlabeled Data = {temp_unlabeled.shape[0]} data')
        self.new_x_data = temp_x_data
        self.new_y_data = temp_y_data
        self.remaining_unlabeled = temp_unlabeled
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        return self.classifier.predict(x_data)