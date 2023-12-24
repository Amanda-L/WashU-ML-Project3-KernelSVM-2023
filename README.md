# Project 3: Kernel SVM
![image](https://github.com/Amanda-L/WashU-ML-Project3-KernelSVM-2023/assets/52643725/81d7fba4-ec15-4848-a47f-def7ff22b695)


In this assignment, the task is to implement a kernel SVM solver using the dual formulation of the SVM optimization problem. Here is a summarized breakdown of the steps involved and files edited:

1. **Spiral Data Set:**
   - Provided with a "spiral" data set loaded as `xTr` and `yTr` in `main.py`.

2. **Kernel Function Implementation (computeK):**
   - Implement the kernel function `computeK(ktype, X, Z, kpar)`:
     - Linear kernel (`ktype='linear'`)
     - Radial basis function (RBF) kernel (`ktype='rbf'`)
     - Polynomial kernel (`ktype='poly'`)
   - Implement the helper function `l2distance(X, Z)` for efficient Euclidean distance calculations.

3. **Quadratic Program Formulation (generateQP):**
   - Implement the function `generateQP(K, yTr, C)` to formulate the quadratic program for the SVM solver.
   - Generate matrices Q, p, G, h, A, and b necessary for the `solve_qp()` solver.

4. **Bias Calculation (recoverBias):**
   - Implement the function `recoverBias(K, yTr, alphas, C)` to calculate the hyperplane bias.

5. **Classifier Creation (createsvmclassifier):**
   - Implement the function `createsvmclassifier(xTr, yTr, alphas, bias)` to create a classifier for SVM.
   - This function defines `svmclassify(xTe)` to classify test data using the trained SVM model.

6. **Training and Evaluation (main.py):**
   - Run `main.py` to create a classifier and print the training error.
   - Comment out code related to functions not yet implemented.
   - The default parameters are C=1, kernel type 'rbf', and polynomial degree P=1.
   - Expected training error with default parameters is 0.04.

7. **Kernel Sensitivity and Cross-Validation:**
   - Implement a function `crossvalidate(xTr, yTr)` to sweep over different values of C and kernel parameters and output the best setting on a validation set.
   - Visualize cross-validation results in `main.py`.
   - Experiment with different parameters for better performance.

The focus is on achieving the lowest training error and optimizing the SVM solver for the given data set. The plotted decision boundary on the training data from
visdecision.py will look like this:
![image](https://github.com/Amanda-L/WashU-ML-Project3-KernelSVM-2023/assets/52643725/cf4c4343-55ea-4b46-8b10-10fa7aeac564)



View 03kernelsvm.html under the Instructions folder for detailed instructions on the assignment.
