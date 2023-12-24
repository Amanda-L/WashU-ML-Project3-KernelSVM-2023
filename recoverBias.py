"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    
    # YOUR CODE HERE
    #find difference between c/2 and all alphas, and the index of the smallest difference (abs)
    diff = abs(alphas - (C/2))
    index = np.argmin(diff)
    alphaY = np.multiply(alphas,yTr)  #shape = (n,1)
    # print(alphaY.shape)
    # print(np.expand_dims(K[index,:], axis=0).shape)
    bias = yTr[index] - np.dot(alphaY.T,np.expand_dims(K[index,:], axis=0).T)
    # print("Bias:",bias[0][0])
    #that's the index that you get y at snd use for the inequality
    
    return bias[0][0]
    

#
# ##TEST recover bias
# K = np.zeros((4,4))
# yTr = np.ones((4,1))
# alphas = np.random.rand(4,1)
# C = 5
# recoverBias(K,yTr,alphas,C)

