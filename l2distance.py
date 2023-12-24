import numpy as np
"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""


def l2distance(X, Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'

    #YOUR CODE HERE
    X_square = np.sum(X ** 2, axis=0, keepdims=True)  # Calculate the sum of squared values for X
    Z_square = np.sum(Z ** 2, axis=0, keepdims=True)  # Calculate the sum of squared values for Z

    cross_term = -2 * np.dot(X.T, Z)  # Calculate the cross term

    D = np.sqrt(X_square.T + cross_term + Z_square)  # Compute the L2 distance

    return D


    return D

# #TEST
# # Create a (3, 4) matrix
# A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# # Create a (3, 3) matrix
# B= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# # Calculate the Euclidean distances between A and B and get the distance matrix D
# result = l2distance(B, A)
#
# # Print the distance matrix D
# print(result)

