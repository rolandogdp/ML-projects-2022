from typing import Tuple
import numpy as np
import matplotlib as plot


def least_squares(y : np.array, tx : np.array) -> Tuple[np.array, float]:
    """Least squares implementation using np.linalg.solver for performance reasons.
    solving: x.T*X *w = X.t*y

    Args:
        y (np.array): Target values
        tx (np.array): Input array x padded with a vector of ones for the bias.

    Returns:
        Tuple[np.array, float]: Weights w as a numpy array, loss value as a float
    """

    #w = np.linalg.inv(tx.T.dot( tx) ).dot(tx.T).dot(y)
    # (y = W*X.T)
    
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    loss = ((y-w.dot(tx.T))**2).mean(axis=0)
   
    return (w,loss)


def ridge_regression(y : np.array, tx: np.array , lambda_: np.array ) -> Tuple[np.array, float]:
    """Ridge regression implementation using linear solver from numpy instead of inverse for performance reasons.
        (X.T*X + lambda*I)*w = X.t*y
    Args:
        y (np.array): The target values Y
        tx (np.array): The input array x padded with a vector of ones for the bias.
        lambda_ (np.array): The regularization parameter

    Returns:
        Tuple[np.array, float]: Weights w as a numpy array, loss value as a float
    """

    w = np.linalg.solve(tx.T.dot(tx) + lambda_*np.ones((1,tx.shape[1])), tx.T.dot(y)) 
    loss = ((y-w.dot(tx.T))**2).mean(axis=0)
    
    
    return (w, loss)

# Testing:
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    np.random.seed = 42
    # x = np.random.random((10,1))
    # w_ori = np.random.random((1,10))

    
    tX = np.array([[1,1, 1], [1,1, 2], [1,2, 2], [1,2, 3]])
    y = np.dot(tX, np.array([2,1, 2])) + 3
    
    
    linreg = LinearRegression().fit(tX,y)
    print(linreg.coef_)
    print(linreg.intercept_)
    w,loss = least_squares(y, tX)
    print(w,loss)
    w_ridge,loss_ridge = ridge_regression(y, tX,10**-5)
    print(w_ridge,loss_ridge)

