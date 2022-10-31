import numpy as np
import matplotlib.pyplot as plt

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
    y_sample = y[random_sample_index]
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
    return -tx.T.dot(e) / y.size


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


def least_squares(y: np.array, tx: np.array):
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


def ridge_regression(y: np.array, tx: np.array, lambda_):
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
    lambda_I = np.eye(D) * np.sqrt(2 * len(y) * lambda_)
    tx_expended = np.append(tx, lambda_I, axis=0)
    y_expended = np.append(y, np.zeros(D))

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
    w, loss = GD_reg(
        y,
        tx,
        initial_w,
        max_iters,
        gamma,
        reg_logistic_gradient,
        logistic_loss,
        lambda_,
    )
    return w, loss


def min_max_normalize(data):
    """Return a min max normalization of the data."""
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def z_normalize(data):
    """Return a z-normalized version of the data."""
    return (data - data.mean(axis=0)) / data.std(axis=0)


def quantile_normalize(data, q=0.75):
    """Return a normalized version of the data using quantiles."""
    low = (1 - q) / 2
    high = 1 - low
    q_low = np.quantile(data, low, axis=0)
    q_high = np.quantile(data, high, axis=0)
    median = np.quantile(data, 0.5, axis=0)
    return (data - median) / (q_high - q_low)


def accuracy(y, y_pred, alpha=0.5, true=1, false=0):
    """Return the accuracy of the model."""
    pred = np.where(y_pred > alpha, true, false)
    correct = np.sum(pred == y)
    return correct / len(y)


# Data Processing:
def build_interaction_tx(input_data, normalisation_function=None):
    """return the input vector tx with interaction terms"""
    # first normalizing the input data
    if not normalisation_function is None:
        input_data = normalisation_function(input_data)

    n_features = input_data.shape[1]
    n_interacted_features = int((n_features - 1) * n_features / 2)

    # creating the future output array
    x = np.empty((n_features + n_interacted_features, len(input_data)))
    x[:n_features] = input_data.T

    # adding interaction predictors to the output array
    index = n_features
    for i in range(n_features):
        for j in range(i):
            x[index] = x[i] * x[j]
            index = index + 1

    # normalizing the data and adding the bias term
    if not normalisation_function is None:
        x = normalisation_function(x.T)
    else:
        x = x.T
    tx = np.append(np.ones(len(x)).reshape(-1, 1), x, axis=1)

    return tx


def balance_classe(y, shuffle=False, seed=42):
    """Returns indices of a random sublist of y, with balanced classes"""
    indices_pos = np.argwhere(y > 0)
    indices_neg = np.argwhere((y < 0) | (y == 0))
    max_len = min((len(indices_pos), len(indices_neg)))
    indices_indices = np.arange(0, max_len, 1)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices_pos)
        np.random.shuffle(indices_neg)

    final_indices = np.concatenate(
        (indices_pos[indices_indices], indices_neg[indices_indices]),
    )
    return final_indices


# Test func
def weighted_logistic_regression(y, tx, initial_w, max_iters, gamma, weights):
    pass

def true_false_positive(y_pred, y_test):
    #source:  https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab
    #TODO: Adapt and optimize
    true_positive = np.equal(y_pred, 1) & np.equal(y_test, 1)
    true_negative = np.equal(y_pred, 0) & np.equal(y_test, 0)
    false_positive = np.equal(y_pred, 1) & np.equal(y_test, 0)
    false_negative = np.equal(y_pred, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    tnr = true_negative.sum() / (true_negative.sum() + false_positive.sum())
    fnr = false_negative.sum() / (false_negative.sum() + true_positive.sum())
    return tpr, fpr, tnr,fnr

def roc_from_scratch(y_pred, y_test, partitions=100):
        #source:  https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab

    roc = np.array([])
    for i in range(partitions + 1):
        
        threshold_vector = np.greater_equal(y_pred, i / partitions).astype(int)
        tpr, fpr, tnr,fnr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr, tnr,fnr])
        
    return roc.reshape(-1, 4)
def auc(roc,partitions,i=0):
    #source:  https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab

    rectangle_roc = 0
    for k in range(partitions):
            rectangle_roc = rectangle_roc + (roc[k, i] - roc[k+1, i]) * roc[k, i+1]
    return rectangle_roc
def plot_scatter(roc,auc=None,id=0,name="test"):
    #source:  https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f' ROC Curve of {name} with auc:{auc}')
    ax1.scatter(roc[:,0],roc[:,1])
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax2.scatter(roc[:,2],roc[:,3])
    
    ax2.set_xlabel('False Negative Rate') 
    ax2.set_ylabel('True Negative Rate')

    plt.ioff()
    fig.savefig("./out/images/"+name+".png")
    plt.close()

def get_and_plot_roc(y_pred,y_test,name="", partitions=100):
    
    roc = roc_from_scratch(y_pred, y_test, partitions=100)
    auc_roc = auc(roc,partitions)
    print(f"Model {name} has auc: {auc_roc}")
    plot_scatter(roc,auc=auc_roc,name=name)
    



    
# K-fold
class K_fold_Iterator:
    # iterator code based on https://stackoverflow.com/questions/21665485/how-to-make-a-custom-object-iterable
    def __init__(self, folds) -> None:
        self.idx = 0
        self.folds = folds
        self.n_splits = len(folds)
        # self.folds_id = list(range(self.n_splits))

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx > self.n_splits:
            raise StopIteration
        test_indx = self.folds.pop(0)
        train_idx = np.concatenate(self.folds)
        self.folds.append(test_indx)
        return test_indx, train_idx


class K_fold:
    def __init__(self, n_splits=5, shuffle=False, random_seed=42) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.splits = None

    def split(self, x, y):
        if self.splits is None:
            np.random.seed(self.random_seed)
            self.x = x
            self.y = y
            if self.shuffle:
                shuffled_indices = np.arange(0, x.shape[0], 1)
                np.random.shuffle(shuffled_indices)
                self.splits = np.array_split(shuffled_indices, self.n_splits)
            else:
                self.splits = np.array_split(np.arange(0, x.shape[0], 1), self.n_splits)

        return K_fold_Iterator(self.splits)

    def __iter__(self):
        if self.splits is None:
            raise ValueError
        return K_fold_Iterator(self.splits)


# Baselines:


class Static_model:
    name = "Static model"
    alpha=0.5
    def __init__(self) -> None:
        pass

    def fit(self, _, y) -> None:

        labels, counts = np.unique(y, return_counts=True)
        self.predominant = labels[np.argmax(counts)]

    def predict(self, x):
        return np.ones(np.shape(x)[0]) * self.predominant


# Models:
class Least_Squares_model:
    name = "Least Squares"
    alpha = 0.35

    def __init__(self) -> None:
        pass

    def fit(self, tx, y) -> None:
        self.w, self.learning_loss = least_squares(y, tx)

    def predict(self, tx) -> np.array:
        return tx.dot(self.w)


class Logistic_Regression_model:
    name = "Logistic Regression"
    alpha = 0.9

    def __init__(self, initial_w, max_iters, gamma) -> None:

        self.initial_w = initial_w
        self.max_iters = max_iters
        self.gamma = gamma

    def fit(self, tx, y) -> None:

        self.w, self.learning_loss = logistic_regression(
            y, tx, self.initial_w, self.max_iters, self.gamma
        )

    def predict(self, tx) -> np.array:
        return tx.dot(self.w)


class Regularized_Logistic_Regression_model(Logistic_Regression_model):
    name = "Regularized Logistic Regression"

    def __init__(self, lambda_, initial_w, max_iters, gamma) -> None:

        super().__init__(initial_w, max_iters, gamma)
        self.lambda_ = lambda_

    def fit(self, tx, y) -> None:
        self.w, self.learning_loss = reg_logistic_regression(
            y, tx, self.lambda_, self.initial_w, self.max_iters, self.gamma
        )


class locally_weighted_logistic_regression(
    object
):  # TODO: This is taking from online for testing purposes, remove this when done TODO
    def __init__(self, tau, reg=0.0001, threshold=1e-6):
        self.reg = reg
        self.threshold = threshold
        self.tau = tau
        self.w = None
        self.theta = None
        self.x = None

    def weights(self, x_train, x):
        sq_diff = (x_train - x) ** 2
        norm_sq = sq_diff.sum(axis=1)
        return np.ravel(np.exp(-norm_sq / (2 * self.tau**2)))

    def logistic(self, x_train):
        return np.ravel(1 / (1 + np.exp(-x_train.dot(self.theta))))

    def train(self, x_train, y_train, x):
        self.w = self.weights(x_train, x)
        self.theta = np.zeros(x_train.shape[1])
        self.x = x
        gradient = np.ones(x_train.shape[1]) * np.inf
        while np.linalg.norm(gradient) > self.threshold:
            # compute gradient
            h = self.logistic(x_train)
            gradient = (
                x_train.T.dot(self.w * (np.ravel(y_train) - h)) - self.reg * self.theta
            )
            # Compute Hessian
            D = np.diag(-(self.w * h * (1 - h)))
            H = x_train.T.dot(D).dot(x_train) - self.reg * np.identity(x_train.shape[1])
            # weight update
            self.theta = self.theta - np.linalg.inv(H).dot(gradient)

    def predict(self):
        return np.array(self.logistic(self.x) > 0.5).astype(int)
