import numpy as np

class Pseudolabel():
    '''
        Pseudolabelling implementation. This algorithm combine labeled data and unlabeled data in training phase.
    '''
    def __init__(self, classifier, sample_rate: float = 0.2):
        self.classifier = classifier
        self.sample_rate = sample_rate
    
    def create_augmentation(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
        num_test_samples = int(len(x_test) * self.sample_rate)
        # train model then create pseudo labels
        self.classifier.fit(x_train, y_train)
        pseudo_labels: np.ndarray = self.classifier.predict(x_test)
        # pick first num_test_samples to concat with training data
        new_y_test = pseudo_labels[num_test_samples:]
        new_x_test = x_test[num_test_samples:, :]
        # concat to new train dataset
        new_x_train = np.vstack([x_train, new_x_test])
        new_y_train = np.append(y_train, new_y_test)
        return new_x_train, new_y_train
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        # save temporary data which contains pseudo-labeled and labeled data
        aug_train_x, aug_train_y = self.create_augmentation(x_train, y_train, x_test)
        self.classifier.fit(aug_train_x, aug_train_y)
    
    def predict(self, x_data: np.ndarray):
        return self.classifier.predict(x_data)