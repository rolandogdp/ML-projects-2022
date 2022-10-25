import numpy as np

def GD(y, tx, initial_w, max_iters, gamma, gradient_func, loss_func):
    """The general Gradient Descent (GD) algorithm."""
    w = initial_w
    
    for i in range(max_iters):
        grad = gradient_func(y, tx, w)
        w = w - gamma * grad

    loss = loss_func(y, tx, w)
    return w, loss


def GD_reg(y, tx, initial_w, max_iters, gamma, gradient_func, loss_func, lambda_):
    """The general Gradient Descent (GD) algorithm with additional regulator lambda."""
    
    w = initial_w
    
    for i in range(max_iters):
        grad = gradient_func(y, tx, w, lambda_)
        w = w - gamma * grad

    loss = loss_func(y, tx, w)
    return w, loss


def get_random_sample(y, tx):
    """get a random sample of (y, tx)"""
    
    random_sample_index = np.random.randint(len(y))
    y_sample  = y [random_sample_index]
    tx_sample = tx[random_sample_index]
    return y_sample, tx_sample


def SGD(y, tx, initial_w, max_iters, gamma, gradient_func, loss_func):
    """The general Stochastic Gradient Descent (SGD) algorithm."""
    w = initial_w
    
    for n_iter in range(max_iters):
        y_sample, tx_sample = get_random_sample(y, tx)
        grad = gradient_func(y_sample, tx_sample, w)
        w = w - gamma * grad
        
    loss = loss_func(y, tx, w)
    return w, loss


def MSE(y, tx, w):
    """Returns the mean square error at w for input tx and output y"""
    e = y - tx.dot(w)
    return np.mean(e**2)


def least_squares_gradient(y, tx, w):
    """Computes the gradient of MSE at w."""
    e = y - tx.dot(w)
    return -tx.T.dot(e)/y.size


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for least squares.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameter as numpy arrays of shape (2, ), for the last iteration of GD
        loss: the loss value corresponding to w
    """
    w, loss = GD(y, tx, initial_w, max_iters, gamma, least_squares_gradient, MSE)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm for least squares using batches of size one.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameter as numpy arrays of shape (2, ), for the last iteration of SGD
        loss: the loss value corresponding to w
    """
    w, loss = SGD(y, tx, initial_w, max_iters, gamma, least_squares_gradient, MSE)
    return w, loss

def least_squares(y : np.array, tx : np.array):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value as a float
    """
    Q, R = np.linalg.qr(tx)
    w = np.linalg.solve(R, Q.T.dot(y))
    loss = MSE(y, tx, w)
    
    return w, loss
    
def ridge_regression(y : np.array, tx: np.array , lambda_):
    """Calculate the least squares solution with regularization parameter.
       returns mse, and optimal weights.
       
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: the regularization parameter as a scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value as a float
    """
    D = tx.shape[1]
    lambda_I = np.eye(D) * np.sqrt(2*len(y)*lambda_)
    tx_expended = np.append(tx, lambda_I, axis=0)
    y_expended  = np.append(y, np.zeros(D))
    
    Q, R = np.linalg.qr(tx_expended)
    w = np.linalg.solve(R, Q.T.dot(y_expended))
    loss = MSE(y, tx, w)
    
    return w, loss
    
def sigmoid(t):
    """Returns sigmoid function on t"""
    exp_t = np.exp(t)
    return exp_t / (1 + exp_t)

def logistic_loss(y, tx, w):
    """Returns the logistic loss at w for input tx and output y"""
    xtw = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(xtw))) - y.T.dot(xtw)
    return np.squeeze(loss)

def logistic_gradient(y, tx, w):
    """Computes the gradient of the logistic loss at w for input tx and output y"""
    return tx.T.dot(sigmoid(tx.dot(w)) - y)

def reg_logistic_gradient(y, tx, w, lambda_):
    """Computes the gradient of the logistic loss with regularizer lambda_ at w for input tx and output y"""
    return logistic_gradient(y, tx, w) + 2 * lambda_ * w
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for logistic regression.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameter as numpy arrays of shape (2, ), for the last iteration of GD
        loss: the loss value corresponding to w
    """
    w, loss = GD(y, tx, initial_w, max_iters, gamma, logistic_gradient, logistic_loss)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for regularized logistic regression.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        lambda_: a scalar denoting the regularization term
        initial_w: numpy array of shape=(D+1, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameter as numpy arrays of shape (2, ), for the last iteration of GD
        loss: the loss value corresponding to w
    """
    w, loss = GD_reg(y, tx, initial_w, max_iters, gamma, reg_logistic_gradient, logistic_loss, lambda_)
    return w, loss

def mim_max_normalize(data):
    """Return a min max normalization of the data."""
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def z_normalize(data):
    """Return a z-normalized version of the data."""
    return (data - data.mean(axis=0)) / data.std(axis=0)

def quantile_normalize(data, q=0.75):
    """Return a normalized version of the data using quantiles."""
    low    = (1-q) / 2
    high   = 1-low
    q_low  = np.quantile(data, low,  axis=0)
    q_high = np.quantile(data, high, axis=0)
    median = np.quantile(data, 0.5, axis=0)
    return (data - median) / (q_high - q_low)
