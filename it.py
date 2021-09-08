import numpy as np
from pyitlib import discrete_random_variable as drv

def accuracy(y_true, y_pred, A=None):
    accuracy = drv.information_mutual(y_true, y_pred, base=np.exp(1))
    return accuracy

def neg_accuracy(y_true, y_pred, A=None):
    neg_accuracy = -accuracy(y_true, y_pred)
    return neg_accuracy

def independence(y, y_pred, A):
        independence = drv.information_mutual(A, y_pred, base=np.exp(1))
        return independence

def balance(y, y_pred, A):
    balance = drv.information_mutual_conditional(y, y_pred, A, base=np.exp(1))
    return balance

def legacy(y, y_pred, A):
    legacy = drv.information_mutual(A, y)
    return legacy

def separation(y, y_pred, A):
    separation = drv.information_mutual_conditional(y_pred, A, y)
    return separation

def sufficiency(y, y_pred, A):
    sufficiency = drv.information_mutual_conditional(y, A, y_pred)
    return sufficiency

def entropy_A(A):
    entropy_A = drv.entropy(A)
    return entropy_A

def entropy_A_Y(A, Y):
    entropy_A_Y = drv.entropy_conditional(A, Y)
    return entropy_A_Y

def entropy_A_R(A, R):
    entropy_A_R = drv.entropy_conditional(A, R)
    return entropy_A_R

def normalized_independence(y, y_pred, A):
    normalized_independence = independence(y, y_pred, A) / entropy_A(A)
    return normalized_independence

def normalized_separation(y, y_pred, A):
    normalized_separation = separation(y, y_pred, A) / entropy_A_Y(A, y)
    return normalized_separation

def normalized_sufficiency(y, y_pred, A):
    normalized_sufficiency = sufficiency(y, y_pred, A) / entropy_A_R(A, y_pred)
    return normalized_sufficiency

def calculate_relevant_metrics(A, Y):
    relevant = {
        'legacy': legacy(Y, None, A),
        'entropy_A': entropy_A(A),
        'entropy_A_Y': entropy_A_Y(A, Y),
    }
    return relevant

it_functions = [neg_accuracy, independence, balance, legacy, separation, sufficiency]