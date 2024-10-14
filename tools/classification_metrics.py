import numpy as np

def calc_mse(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate Mean Square Error (MSE) for Linear Regression

        Input: list of predictions, y_test data

        Output: mse result
    '''
    subs = np.subtract(y_test, preds)
    mse = np.mean(np.square(subs))
    return round(mse, 3)

def calc_mae(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate Mean Average Error (MAE) for Linear Regression

        Input: list of predictions, y_test data
        
        Output: mae result
    '''
    subs = np.subtract(y_test, preds)
    mae = np.mean(np.abs(subs))
    return round(mae, 3)

def calc_r2_score(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function to calculate R-Squared Error for Linear Regression

        Input: list of predictions, y_test data
        
        Output: r2-square result
    '''
    sse = sum((preds - y_test)**2)
    tse = (len(y_test) - 1) * np.var(y_test, ddof=1)
    r2_score = np.mean(1 - (sse / tse))
    return r2_score

def calc_accuracy(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function for calculate accuracy of prediction

        Input: list of predictions, y_test data

        Output: accuracy result
    '''
    acc = np.mean(np.equal(preds, y_test))
    return round(acc * 100, 3)

def calc_error_rate(preds: np.ndarray, y_test: np.ndarray) -> float:
    '''
        Function for calculate error rate in prediction

        Input: list of predictions, y_test data

        Output: error rate result
    '''
    error_rate = np.mean(np.not_equal(preds, y_test))
    return round(error_rate * 100, 3)

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

def calc_specificity(true_negative: float, false_positive: float) -> float:
    '''
        Function for calculate specificity with equation -> recall = TN / (TN + FP)

        Input: true negative value, false positive value

        Output: specificity result
    '''
    return round(true_negative / (true_negative + false_positive), 3)

def calc_f1score(preds: np.ndarray, y_test: np.ndarray):
    '''
        Function for calculate f1-score with equation -> f1-score = (2 * precision * recall) / (precision + recall)

        Input: prediction result, y test result

        Output: f1-score result
    '''
    true_positive, true_negative, false_positive, false_negative = np.ravel(calc_confusion_matrix(preds, y_test))
    # special case of f1 score
    if true_positive == 0 and false_negative == 0 and false_positive == 0:
        precision_score = 1
        recall_score = 1
        specificity_score = 1
        f1_score = 1
    elif false_negative == 0 or false_positive == 0 or true_positive == 0:
        precision_score = 0
        recall_score = 0
        specificity_score = 0
        f1_score = 0
    else:
        precision_score = calc_precision(false_positive, true_positive)
        recall_score = calc_recall(false_negative, true_positive)
        specificity_score = calc_specificity(true_negative, false_positive)
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)
    return round(precision_score, 3), round(recall_score, 3), round(specificity_score, 3),round(f1_score, 3)

def calc_confusion_matrix(preds: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    '''
        Function for calculate confusion matrix in order [TN, FP], [FN, TP]
    
        Input: prediction result, y test result

        Output: confusion matrix in array shape
    '''
    true_negative = np.sum(np.bitwise_and(np.equal(y_test, 0), np.equal(preds, 0)))
    true_positive = np.sum(np.bitwise_and(np.equal(y_test, 1), np.equal(preds, 1)))
    false_negative = np.sum(np.bitwise_and(np.equal(y_test, 1), np.equal(preds, 0)))
    false_positive = np.sum(np.bitwise_and(np.equal(y_test, 0), np.equal(preds, 1)))
    return np.array([[true_positive, true_negative], [false_positive, false_negative]])

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
        precision, recall, specificity, f1score = calc_f1score(preds, y_test)
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'Specificity = {specificity}')
        print(f'F1-Score = {f1score}')
        print(f'Confusion Matrix =\n{calc_confusion_matrix(preds, y_test)}')
    else: 
        print(f'Evaluation Report For {algo} Algorithm')
        print(f'MSE = {calc_mse(preds, y_test)}')
        print(f'RMSE = {round(np.sqrt(calc_mse(preds, y_test)), 3)}')
        print(f'MAE = {calc_mae(preds, y_test)}')
        print(f'R2 Square = {calc_r2_score(preds, y_test)}')
