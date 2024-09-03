import numpy as np

'''
    Function to calculate Mean Square Error (MSE)

    Input: list of predictions, y_test data
    Output: mse in percent
'''
def calc_mse(preds: np.ndarray, y_test: np.ndarray) -> float:
    mse = np.mean(np.square(y_test - preds))
    return round(mse, 3)

'''
    Function for calculate accuracy of prediction

    Input: list of predictions, y_test data
    Output: accuracy value in percent
'''
def calc_accuracy(preds: np.ndarray, y_test: np.ndarray) -> float:
    acc = np.sum(np.equal(preds, y_test)) / len(y_test)
    return round(acc, 3)

'''
    Function for calculate error rate in prediction

    Input: list of predictions, y_test data
    Output: error rate value in percent
'''
def calc_error_rate(preds: np.ndarray, y_test: np.ndarray) -> float:
    error_rate = np.sum(np.not_equal(preds, y_test)) / len(y_test)
    return round(error_rate, 3)

'''
    Function for calculate precision with equation -> precision = TP / (TP + FP)

    Input: false positive value, true positive value
    Output: precision result
'''
def calc_precision(false_positive: float, true_positive: float) -> float:
    return round(true_positive / (true_positive + false_positive), 3)

'''
    Function for calculate recall with equation -> recall = TP / (TP + FN)

    Input: false negative value, true positive value
    Output: recall result
'''
def calc_recall(false_negative: float, true_positive: float) -> float:
    return round(true_positive / (true_positive + false_negative), 3)

'''
    Function for calculate f1-score with equation -> f1-score = (2 * precision * recall) / (precision + recall)

    Input: prediction result, y test result
    Output: f1 score result
'''
def calc_f1score(preds: np.ndarray, y_test: np.ndarray) -> float:
    _, false_positive, false_negative, true_positive = calc_confusion_matrix(preds, y_test).ravel()
    precision_score = calc_precision(false_positive, true_positive)
    recall_score = calc_recall(false_negative, true_positive)
    print(f'Precision = {precision_score}')
    print(f'Recall = {recall_score}')
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)
    return round(f1_score, 3)

'''
    Function for calculate confusion matrix in order [TN, FP], [FN, TP]
'''
def calc_confusion_matrix(preds: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    true_negative = ((y_test == 0) & (preds == 0)).sum()
    true_positive = ((y_test == 1) & (preds == 1)).sum()
    false_negative = ((y_test == 1) & (preds == 0)).sum()
    false_positive = ((y_test == 0) & (preds == 1)).sum()
    return np.array([[true_negative, false_positive], [false_negative, true_positive]])

'''
    Function for report the evaluation result of supervised algorithm
'''
def evaluation_report(algo: str, preds: np.ndarray, y_test: np.ndarray) -> float:
    print(f'Evaluation Report For {algo} Algorithm')
    print(f'Accuracy = {calc_accuracy(preds, y_test)} %')
    print(f'Error Rate = {calc_error_rate(preds, y_test)} %')
    print(f'F1-Score = {calc_f1score(preds, y_test)}')
    print(f'Confusion Matrix =\n{calc_confusion_matrix(preds, y_test)}')

def compare_evaluation():
    return 0