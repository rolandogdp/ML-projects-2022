from typing import Tuple
import numpy as np
import matplotlib as plot


def least_squares(y : np.array, tx : np.array) -> Tuple[np.array, int]:

    #w = np.linalg.inv(tx.T.dot( tx) ).dot(tx.T).dot(y)
    w = np.linalg.solve(tx,y)
    loss = (y-w.dot(tx)).mean(ax=0)

    return (w,loss)


def ridge_regression(y : np.array, tx: np.array , lambda_: np.array ) -> Tuple[np.array, int]:
    N= y.shape[0]
    w = np.linalg.inv(tx.T.dot(tx) + (lambda_/(2*N))).dot(tx.T).dot(y)
    loss = ((y-w.dot(tx))).mean(ax=0)
    return (w, loss)

# Testing:
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    x = [12,16,71,99,45,27,80,58,4,50]
    y = [56,22,37,78,83,55,70,94,12,40]
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    linreg = LinearRegression().fit(x,y)
    
    w,loss = least_squares(y, x)

