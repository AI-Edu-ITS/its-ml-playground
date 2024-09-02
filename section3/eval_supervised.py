import numpy as np

'''
    Function for calculate accuracy of prediction

    Input: list of predictions, y_test data
    Output: accuracy value in percent
'''
def calc_accuracy(preds: np.ndarray, y_test: np.ndarray) -> float:
    return 100 * (preds==y_test).mean()