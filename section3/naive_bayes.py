import numpy as np

class GaussianNaiveBayes():
    '''
        Naïve Bayes Class using Gaussian. This algorithm based on Bayes Theorem which described as same as equation below:
        
        P(y|x1, ...xj) = P(x1, ...xj|y) P(y) / P(x1,...,xj) where x1, ...xj represent features that are independent (data), while y dependent variable (target)
        
        This can be simplified as equation below:
        
        Posterior Probability = (Likelihood * Prior Probability) / Predictor Prior Probability

        Because we use Gaussian Naive Bayes, we implement P(x1, ...xj|y) or Likelihood with equation as below:

        a = 1/sqrt(2 * pi * sigma^2)
        
        b = exp(-((x - mu)^2) / 2 * sigma^2)
        
        P(x1, ...xj|y) = (a * b)
    '''
    def __init__(self):
        self.mean: np.ndarray
        self.variance = np.ndarray
        self.priors = np.ndarray
        self.classes = np.ndarray
        self.num_classes = int

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = x_train.shape
        self.classes = np.bincount(y_train)
        self.num_classes = len(self.classes)
        # define new empty-zero matrix with size num_classes x num_features and it should store float data type 32 bit
        self.mean = np.zeros((self.num_classes, n_features), dtype=np.float32)
        self.variance = np.zeros((self.num_classes, n_features), dtype=np.float32)
        # define new empty-zero matrix for store prior probability value 
        self.priors = np.zeros(self.num_classes, dtype=np.float32)

        for i, c in enumerate(self.classes):
            temp_x_class = x_train[np.equal(y_train, c)]
            self.mean[i, :] = temp_x_class.mean(axis=0)
            self.variance[i, :] = temp_x_class.var(axis=0)
            self.priors[i] = temp_x_class.shape[0] / float(n_samples)

    def calc_gaussian_likelihood(self, class_idx: int, x_test: np.ndarray) -> np.array:
        a = 1 / (np.sqrt(2* np.pi * self.variance[class_idx]))
        b = np.exp(-(x_test - self.mean[class_idx]) ** 2 / (2 * self.variance[class_idx]))
        return a * b

    def calc_posterior_probability(self, x_test: np.ndarray):
        posterior_list = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self.calc_gaussian_likelihood(i, x_test)))
            posterior_list.append(prior + posterior)
        return self.classes[np.argmax(posterior_list)]
    
    def predict(self, x_test: np.ndarray) -> np.array:
        preds = [self.calc_posterior_probability(x) for x in x_test]
        return np.array(preds)
