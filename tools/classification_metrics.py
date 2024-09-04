import numpy as np

def calc_mse(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate Mean Square Error (MSE) for Linear Regression

        Input: list of predictions, y_test data

        Output: mse result
    '''
    mse = np.mean(np.square(y_test - preds))
    return round(mse, 3)

def calc_mae(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate Mean Average Error (MAE) for Linear Regression

        Input: list of predictions, y_test data
        
        Output: mape result
    '''
    mape = np.mean(np.abs((y_test - preds)))
    return round(mape, 3)

def calc_r2_square(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate R-Squared Error for Linear Regression. this function needs
        to calculate Sum of Squares of the Residuals (SSres) and calculate Sum of Squares Total (SStot)

        Input: list of predictions, y_test data
        
        Output: r2-square result
    '''
    y_test_mean = y_test.mean()
    ssres = ((y_test - preds) ** 2).sum()
    sstot = ((y_test - y_test_mean) ** 2).sum()
    r2_square = 1 - (ssres/sstot)
    return round(r2_square, 3)

def calc_accuracy(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function for calculate accuracy of prediction

        Input: list of predictions, y_test data

        Output: accuracy result
    '''
    acc = np.sum(np.equal(preds, y_test)) / len(y_test)
    return round(acc, 3)

def calc_error_rate(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function for calculate error rate in prediction

        Input: list of predictions, y_test data

        Output: error rate result
    '''
    error_rate = np.sum(np.not_equal(preds, y_test)) / len(y_test)
    return round(error_rate, 3)

def calc_precision(false_positive: float, true_positive: float) -> float:
    '''
        Function for calculate precision with equation -> precision = TP / (TP + FP)

        Input: false positive value, true positive value

        Output: precision result
    '''
    return round(true_positive / (true_positive + false_positive), 3)

def calc_recall(false_negative: float, true_positive: float) -> float:
    '''
        Function for calculate recall with equation -> recall = TP / (TP + FN)

        Input: false negative value, true positive value

        Output: recall result
    '''
    return round(true_positive / (true_positive + false_negative), 3)

def calc_f1score(preds: np.ndarray, y_test: np.ndarray):
    '''
        Function for calculate f1-score with equation -> f1-score = (2 * precision * recall) / (precision + recall)

        Input: prediction result, y test result

        Output: f1-score result
    '''
    _, false_positive, false_negative, true_positive = calc_confusion_matrix(preds, y_test).ravel()
    # special case of f1 score
    if true_positive == 0 and false_negative == 0 and false_positive == 0:
        precision_score = 1
        recall_score = 1
        f1_score = 1
    elif false_negative == 0 or false_positive == 0 and true_positive == 0:
        precision_score = 0
        recall_score = 0
        f1_score = 0
    else:
        precision_score = calc_precision(false_positive, true_positive)
        recall_score = calc_recall(false_negative, true_positive)
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)
    return precision_score, recall_score, round(f1_score, 3)

def calc_confusion_matrix(preds: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    '''
        Function for calculate confusion matrix in order [TN, FP], [FN, TP]
    
        Input: prediction result, y test result

        Output: confusion matrix in array shape
    '''
    true_negative = ((y_test == 0) & (preds == 0)).sum()
    true_positive = ((y_test == 1) & (preds == 1)).sum()
    false_negative = ((y_test == 1) & (preds == 0)).sum()
    false_positive = ((y_test == 0) & (preds == 1)).sum()
    return np.array([[true_negative, false_positive], [false_negative, true_positive]])

def evaluation_report(algo: str, preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function for report the evaluation result of supervised algorithm

        Input: algorithm name, prediction result, y test result
        
        Output: console value of each metric
    '''
    if 'regression' not in algo:
        print(f'Evaluation Report For {algo} Algorithm')
        print(f'Accuracy = {calc_accuracy(preds, y_test)} %')
        print(f'Error Rate = {calc_error_rate(preds, y_test)} %')
        precision, recall, f1score = calc_f1score(preds, y_test)
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1-Score = {f1score}')
        print(f'Confusion Matrix =\n{calc_confusion_matrix(preds, y_test)}')
    else: 
        print(f'Evaluation Report For {algo} Algorithm')
        print(f'MSE = {calc_mse(preds, y_test)}')
        print(f'RMSE = {round(np.sqrt(calc_mse(preds, y_test)), 3)}')
        print(f'MAE = {calc_mae(preds, y_test)}')
        print(f'R2 Square = {calc_r2_square(preds, y_test)}')

def compare_evaluation():
    return 0