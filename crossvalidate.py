"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    
    # YOUR CODE HERE
    # Load data
    # xTr = np.genfromtxt('xTr.csv', delimiter=',')
    # yTr = np.genfromtxt('yTr.csv', delimiter=',').reshape((xTr.shape[1], 1))

    for P in range(len(paras)):
        for C in range(len(Cs)):
            svmclassify = trainsvm(xTr, yTr, Cs[C], ktype, paras[P])
            train_preds = svmclassify(xTr)
            train_error = np.mean(train_preds != yTr)
            errors[P,C] = train_error
    ind1, ind2 = np.unravel_index(errors.argmin(), errors.shape)
    bestC = Cs[ind2]
    bestP = paras[ind1]
    lowest_error = errors[ind1,ind2]

    # print(":", train_error)

    
    return bestC, bestP, lowest_error, errors


# #TEST CROSSVALIDATE
# xTr = np.genfromtxt('xTr.csv', delimiter=',')
# yTr = np.genfromtxt('yTr.csv', delimiter=',').reshape((xTr.shape[1], 1))
# Cs = [1, 10]
# Ps = [1, 10]
# crossvalidate(xTr, yTr, 'rbf', Cs, Ps)