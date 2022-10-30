import numpy as np
import pandas as pd
import argparse
import random

import helpers
import implementation

def accuracy(y, tx, w):
    """Return the accuracy of the model."""
    pred    = np.where(tx.dot(w) > 0, 1, 0)
    correct = np.sum(pred == y)
    return correct / len(y)

def build_test_train(y, tx, ratio=0.9, seed=1):
    """Split the dataset (y, tx) into training/testing set according to the split ratio"""
    # performing permutation before splitting the dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(y))

    # defining indices for y, tx
    delimiter_indice = int(ratio * len(y))
    te_indices = indices[delimiter_indice:]
    tr_indices = indices[:delimiter_indice]

    # creating the train/test sets
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    tx_te = tx[te_indices]
    tx_tr = tx[tr_indices]
    return y_te, y_tr, tx_te, tx_tr

def main(config):
    yb,      input_data,      ids      = helpers.load_csv_data("./data/train.csv")
    yb_test, input_data_test, ids_test = helpers.load_csv_data("./data/test.csv")
    # creating classification vector y that fits for logistic regression
    y  = np.where(yb > 0, 1, 0)
    # Preprocessing the input data:
    preprocessing_methods = [implementation.z_normalize,implementation.quantile_normalize,implementation.min_max_normalize]
    x = preprocessing_methods[config.preprocessing](input_data)
    tx = np.append(np.ones(len(x)).reshape(-1,1), x, axis=1)

    #Baseline models
    # TODO: add baselines
    # Ideas: most frequent class, simple bayesian model Prior? Constant?
    

    # Train model:
    w_log_reg, loss = implementation.logistic_regression(y, tx, initial_w=np.zeros(tx.shape[1]), max_iters=2000, gamma=0.000003)
    w_ls, loss      = implementation.least_squares(yb, tx)


    #
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', type=int, default=42, help='Random number generator seed')     # for randomness; initially None
    # Run
    # Data        
    # Model
    parser.add_argument('--model_name', type=str, default="", help='Default model name to load')          
    # Learning args
    parser.add_argument('--preprocessing', type=int, default=0, help='Index from : [implementation.z_normalize,implementation.quantile_normalize,implementation.min_max_normalize]')
    parser.add_argument('--train_val_ratio', type=float, default=0.95, help='The training/validation ratio to use for the given dataset')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of train epochs')
    parser.add_argument('--k_fold', type=int, default=0, help='Number of folds for K-fold validation ')    
    config = parser.parse_args()

    # Reproducibility:
    np.random.seed(config.seed)
    random.seed(config.seed)


    main(config)
