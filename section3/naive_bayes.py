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
        self.mean = np.ndarray
        self.variance = np.ndarray
        self.priors = np.ndarray
        self.classes = np.ndarray
        self.num_classes = int

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = x_train.shape
        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)
        # define new empty-one matrix with size num_classes x num_features and it should store float data type 32 bit
        self.mean = np.ones((self.num_classes, n_features))
        self.variance = np.ones((self.num_classes, n_features))
        # define new empty-one matrix for store prior probability value 
        self.priors = np.ones(self.num_classes)

        for i, c in enumerate(self.classes):
            temp_x_class = x_train[np.equal(y_train, c)]
            self.mean[i, :] = np.mean(temp_x_class, axis=0)
            self.variance[i, :] = np.var(temp_x_class,axis=0)
            self.priors[i] = temp_x_class.shape[0] / n_samples

    def calc_gaussian_likelihood(self, class_idx: int, x_data: np.ndarray) -> np.array:
        a = 1 / (np.sqrt(2* np.pi * self.variance[class_idx]))
        b = np.exp(-(x_data - self.mean[class_idx]) ** 2 / (2 * self.variance[class_idx]))
        return a * b

    def calc_posterior_probability(self, x_test: np.ndarray) -> np.ndarray:
        posterior_list = np.zeros((len(self.classes)))
        for idx in self.classes:
            # adding 1 in log operation to prevent negative value
            prior = np.log(1 + self.priors[idx])
            posterior = np.log(1 + self.calc_gaussian_likelihood(idx, x_test))
            posterior_list[idx] = prior + np.prod(posterior)
        return posterior_list
    
    def predict(self, x_test: np.ndarray) -> np.array:
        preds = np.zeros((len(x_test)), dtype=np.uint32)
        for x_idx, x_data in enumerate(x_test):
            res = self.calc_posterior_probability(x_data)
            preds[x_idx] = self.classes[np.argmax(res)]
        return preds

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        preds_proba = np.zeros((len(x_test), self.num_classes))
        for x_idx, x_data in enumerate(x_test):
            res = self.calc_posterior_probability(x_data)
            for res_idx, res_data in enumerate(res):
                preds_proba[x_idx, res_idx] = res_data
        return preds_proba