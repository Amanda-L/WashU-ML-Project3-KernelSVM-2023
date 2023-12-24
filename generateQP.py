"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE
    # Hessian matrix
    Q = np.outer(yTr, yTr) * K

    # p vector, negative because of subtraction
    p = -np.ones((1,n))
   # Inequality constraints
    G = np.vstack((np.eye(n),-1*np.eye(n)))  # Identity matrix
    h = np.vstack((C*np.ones((n,1)), np.zeros((n,1))))

    # Equality constraints
    A = yTr.T
    b = np.zeros((1,1))

    return Q, p, G, h, A, b

            

