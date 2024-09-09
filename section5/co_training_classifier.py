import copy
import numpy as np
import random

class CoTrainingClassifier():
    '''
        Function for Co Training Classification. You can use any supervised approach with this classifier 
        such as logistic regression, svm, knn, naïve bayes, etc.
    '''
    def __init__(
        self, 
        classifier, 
        iterations: int = 100, 
        positive_samples: int = -1, 
        negative_samples: int = -1,
        unlabeled_samples: int = 75
    ):
        self.classifier_1 = classifier
        self.classifier_2 = copy.copy(classifier)
        self.epochs = iterations
        self.pos_samples = positive_samples
        self.neg_samples = negative_samples
        self.unlabel_samples = unlabeled_samples
    
    def fit(self, x_data_1: np.ndarray, x_data_2: np.ndarray, y_data: np.ndarray):
        # init variables
        cur_iter = 0
        # check negative and positive samples
        if self.pos_samples == -1 and self.neg_samples == -1:
            num_pos = sum(1 for y_i in y_data if y_i == 1)
            num_neg = sum(1 for y_i in y_data if y_i == 0)
            ratio  = num_neg / float(num_pos)
			
            if ratio  > 1:
                self.pos_samples = 1
                self.neg_samples = round(self.pos_samples * ratio)
            else:
                self.pos_samples = round(self.neg_samples/ratio)
                self.neg_samples = 1
        unlabeled = [i for i, y_i in enumerate(y_data) if y_i == -1]
        # randomize unlabeled data
        random.shuffle(unlabeled)
        temp_unlabeled = unlabeled[-min(len(unlabeled), self.unlabel_samples):]
        # partial labeled samples
        labeled = [i for i, y_i in enumerate(y_data) if y_i != -1]
        # remove samples from unlabeled
        unlabeled = unlabeled[:-len(temp_unlabeled)]
        while cur_iter != self.epochs and unlabeled:
            cur_iter += 1
            self.classifier_1.fit(x_data_1[labeled], y_data[labeled])
            self.classifier_2.fit(x_data_2[labeled], y_data[labeled])

            # predict probability of each class
            y1_data_prob: np.ndarray = self.classifier_1.predict_proba(x_data_1[temp_unlabeled])
            y2_data_prob: np.ndarray = self.classifier_2.predict_proba(x_data_2[temp_unlabeled])

            # negative and positive list
            neg_list, pos_list = [], []
            for i in (np.argsort(y1_data_prob[:,0]))[-self.neg_samples:]:
                if y1_data_prob[i,0] > 0.5:
                    neg_list.append(i)
            for i in (np.argsort(y1_data_prob[:,0]))[-self.pos_samples:]:
                if y1_data_prob[i,0] > 0.5:
                    pos_list.append(i)

            for i in (np.argsort(y2_data_prob[:,0]))[-self.neg_samples:]:
                if y2_data_prob[i,0] > 0.5:
                    neg_list.append(i)
            for i in (np.argsort(y2_data_prob[:,0]))[-self.pos_samples:]:
                if y2_data_prob[i,0] > 0.5:
                    pos_list.append(i)

			# label the samples and remove thes newly added samples from temp_unlabeled
            y_data[[temp_unlabeled[x] for x in pos_list]] = 1
            y_data[[temp_unlabeled[y] for y in neg_list]] = 0
            labeled.extend([temp_unlabeled[x] for x in pos_list])
            labeled.extend([temp_unlabeled[y] for y in neg_list])
            temp_unlabeled = [i for i in temp_unlabeled if not (i in pos_list or i in neg_list)]

			# add new elements to U_
            temp_counter = 0
            while temp_counter != (len(pos_list) + len(neg_list)) and unlabeled:
                temp_counter += 1
                temp_unlabeled.append(unlabeled.pop())
        
        # train final model
        self.classifier_1.fit(x_data_1[labeled], y_data[labeled])
        self.classifier_2.fit(x_data_2[labeled], y_data[labeled])

    def is_support_proba(self, classifier, x_data: np.ndarray) -> bool:
        try:
            classifier.predict_proba(x_data)
            return True
        except:
            return False
        
    def predict(self, x_data_1: np.ndarray, x_data_2: np.ndarray) -> np.ndarray:
        # predict data
        y1_data = self.classifier_1.predict(x_data_1)
        y2_data = self.classifier_2.predict(x_data_2)
        # check is predict probability supported or not
        is_proba_supported = self.is_support_proba(self.classifier_1, x_data_1) and self.is_support_proba(self.classifier_2, x_data_2)
        y_preds = np.asarray([-1] * x_data_1.shape[0])

        for idx, (y1_i, y2_i) in enumerate(zip(y1_data, y2_data)):
            if y1_i == y2_i:
                y_preds[idx] = y1_i
            elif is_proba_supported:
                y1_proba = self.classifier_1.predict_proba([x_data_1[idx]])[0]
                y2_proba = self.classifier_2.predict_proba([x_data_2[idx]])[0]
                sum_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_proba, y2_proba)]
                y_preds[idx] = sum_probs.index(max(sum_probs))
            else:
                # if predict proba not supported
                y_preds[idx] = random.randint(0,1)
        
        return y_preds
    
    def predict_proba(self, x_data_1: np.ndarray, x_data_2: np.ndarray):
        y_proba = np.full((x_data_1.shape[0], 2), -1, dtype=np.float64)
        y1_proba = self.clf1_.predict_proba(x_data_1)
        y2_proba = self.clf2_.predict_proba(x_data_2)

        for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
            y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
            y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

        return y_proba