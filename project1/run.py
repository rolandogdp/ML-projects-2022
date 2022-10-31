import numpy as np
import argparse
import random
import pickle
import helpers
import implementations


def accuracy(y, tx, w):
    """Return the accuracy of the model."""
    pred = np.where(tx.dot(w) > 0, 1, 0)
    correct = np.sum(pred == y)
    return correct / len(y)


def build_test_train(y, tx, ratio=0.9, seed=42):
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


def cross_validate_model(
    model, x, y, k_fold=None, k_folds=5, random_seed=42, has_train_loss=False
):

    if k_fold is None:
        k_fold = implementations.K_fold(k_folds, shuffle=True, random_seed=random_seed)
    accuracies = []
    if has_train_loss:
        train_losses = []
    it = 0
    for train_index, test_index in k_fold.split(x, y):
        x_train, x_test, y_train, y_test = (
            x[train_index],
            x[test_index],
            y[train_index],
            y[test_index],
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        implementations.get_and_plot_roc(y_pred,y_test,name="K_fold_"+model.name+str(it),partitions=100)
        accu = implementations.accuracy(y_test, y_pred,model.alpha)
        accuracies.append(accu)
        it+=1
        if has_train_loss:
            train_losses.append(model.learning_loss)
    if has_train_loss:
        return accuracies, train_losses
    return accuracies


def calculate_statistics(accuracies):
    print(accuracies)
    array = np.array(accuracies)
    mean = np.mean(array)
    std = np.std(array)
    # Should we also calcute percentiles ? 95%?
    return mean, std


def print_statistics(stats, name: str):
    print(f"{name} has mean accuracy: {stats[0]} +- {stats[1]} std")


def evaluate_model(model, tx, y, k_fold, name="", has_train_loss=False):
    print(f"Starting Evaluation of model: {name}")
    if has_train_loss:
        test_losses, train_losses = cross_validate_model(
            model, tx, y, k_fold=k_fold, has_train_loss=has_train_loss
        )
        statistics_test = calculate_statistics(test_losses)
        statistics_train = calculate_statistics(train_losses)
        print_statistics(statistics_train, name=name + " Training")
        print_statistics(statistics_test, name=name + " Test")
        return statistics_test, statistics_train
    else:
        test_losses = cross_validate_model(model, tx, y, k_fold=k_fold)
        statistics = calculate_statistics(test_losses)
        print_statistics(statistics, name=name)
        return statistics


def main(config):
    print("Loading Data")
    yb, input_data, ids = helpers.load_csv_data("./data/train.csv")
    y = np.where(yb > 0, 1, 0)
    # Preprocessing the input data:
    print("Preprocessing Data..")
    preprocessing_methods = [
        lambda x: x,
        implementations.z_normalize,
        implementations.quantile_normalize,
        implementations.min_max_normalize,
    ]
    x = preprocessing_methods[config.preprocessing](input_data)
    tx = np.append(np.ones(len(x)).reshape(-1, 1), x, axis=1)

    print("Creating Models..")
    # Baseline models
    # TODO: add baselines
    # TODO:Ideas: most frequent class, simple bayesian model Prior? Constant?
    
    static_model = implementations.Static_model()

    # Defining models:
    logistic_regression = implementations.Logistic_Regression_model(
        initial_w=np.zeros(tx.shape[1]), max_iters=10000, gamma=0.1
    )
    least_squares = implementations.Least_Squares_model()

    # Cross validating models:

    print("Crossvalidating Models..")
    # With default non processed data
    k_fold = implementations.K_fold(
        5
    )  # Reusing same k_folds for performance and comparability reasons.

    static_model_res = evaluate_model(
        static_model, tx, y, k_fold=k_fold, name="static model", has_train_loss=False
    )
    logistic_regression_res = evaluate_model(
        logistic_regression,
        tx,
        y,
        k_fold=k_fold,
        name="logistic regression",
        has_train_loss=True,
    )
    least_squares_res = evaluate_model(
        least_squares, tx, y, k_fold=k_fold, name="least squares", has_train_loss=True
    )

    # With Z-score processed data:
    print("Cross Validation With Z-score processed data")
    tx_z_normalized = implementations.z_normalize(tx)

    k_fold_normalized = implementations.K_fold(
        5
    )  # Reusing same k_folds for performance and comparability reasons.

    logistic_regression = implementations.Logistic_Regression_model(
        initial_w=np.zeros(tx_z_normalized.shape[1]), max_iters=10000, gamma=0.1
    )

    static_model_res = evaluate_model(
        static_model, tx_z_normalized, y, k_fold=k_fold_normalized, name="static model"
    )
    logistic_regression_res = evaluate_model(
        logistic_regression,
        tx_z_normalized,
        y,
        k_fold=k_fold_normalized,
        name="logistic regression",
        has_train_loss=True,
    )
    least_squares_res = evaluate_model(
        least_squares,
        tx_z_normalized,
        y,
        k_fold=k_fold_normalized,
        name="least squares",
        has_train_loss=True,
    )

    # With Interaction Matrix:
    print("Cross Validation With Interaction Matrix this time:")
    tx_interactions = implementations.build_interaction_tx(
        input_data, implementations.z_normalize
    )

    k_fold_interactions = implementations.K_fold(
        5
    )  # Reusing same k_folds for performance and comparability reasons.

    logistic_regression = implementations.Logistic_Regression_model(
        initial_w=np.zeros(tx_interactions.shape[1]), max_iters=2000, gamma=0.000003
    )

    static_model_res = evaluate_model(
        static_model,
        tx_interactions,
        y,
        k_fold=k_fold_interactions,
        name="static model",
    )
    logistic_regression_res = evaluate_model(
        logistic_regression,
        tx_interactions,
        y,
        k_fold=k_fold_interactions,
        name="logistic regression",
        has_train_loss=True,
    )
    least_squares_res = evaluate_model(
        least_squares,
        tx_interactions,
        y,
        k_fold=k_fold_interactions,
        name="least squares",
        has_train_loss=True,
    )

    # Analyzing Model Performances:
    # With default non processed data:

    # Printing:

    # Plotting..?
    # TODO
    # Running Best model:

    #TODO: Uncomment code below
    print("Running Cross validation on best model ")
    best_model = implementations.Logistic_Regression_model(
        initial_w=np.zeros(tx_interactions.shape[1]), max_iters=8000, gamma=0.0000005
    )
    best_model_res = evaluate_model(
        best_model,
        tx_interactions,
        y,
        k_fold=k_fold_interactions,
        name="Best model LogReg",
    )

    # Training on whole dataset:
    
    try:
        print("Trying to load best model:")
        best_model_1 = pickle.load(open("./out/best_model_1.pickle", "rb"))
        best_model_1.name = "Best_model"

    except (OSError, IOError) as e:
        print("Best model pickle unavailable, creating new one")
        best_model_1 = implementations.Logistic_Regression_model(
            initial_w=np.zeros(tx_interactions.shape[1]), max_iters=8000, gamma=0.0000005
        )
        best_model_1.name = "Best_model"
        print("Training Best model on full dataset:")
        best_model_1.fit(tx_interactions, y)

        pickle.dump(best_model_1, open("./out/best_model_1.pickle", "wb"))

    
    loss = best_model_1.learning_loss
    print(f"Training loss of best: {loss}")
    y_train_pred = best_model_1.predict(tx_interactions)
    tains_accs = []
    alphas = list(np.arange(0,1,0.1))
    for alpha in alphas:
        train_acc = implementations.accuracy(y_train_pred,y,alpha=alpha)
        print(f"train_acc = {train_acc} with alpha = {alpha} ")
        tains_accs.append(train_acc)
    print(f"Training accuracy of Best model mean:{np.mean(tains_accs)}\
         and we have max accuracy of { tains_accs[np.argmax(tains_accs)]} with alpha= { alphas[np.argmax(tains_accs)]}")
    implementations.get_and_plot_roc(y_train_pred,y,name="Best_model_traindata",partitions=100)

    print("Loading Leaderboard test data")
    _, input_data_test, ids_test = helpers.load_csv_data("./data/test.csv")
    # creating classification vector y that fits for logistic regression

    tx_test = implementations.build_interaction_tx(
        input_data_test, implementations.z_normalize
    )

    print("Learning Leaderboard test data")
    y_test_pred = best_model_1.predict(tx_test)

    print("Predicting:")
    y_pred = np.where(y_test_pred > 0.5, 1, -1)

    name = "./out/submission.csv"
    print(f"Saving Prediction file to {name}")
    helpers.create_csv_submission(ids_test, y_pred, name)


# TODO

# TODO
#


if __name__ == "__main__":
    # TODO: Clean this arguments, keep only what's neeeded
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument(
        "--seed", type=int, default=42, help="Random number generator seed"
    )  # for randomness; initially None
    # Run
    # Data
    # Model
    parser.add_argument(
        "--model_name", type=str, default="", help="Default model name to load"
    )
    # Learning args
    parser.add_argument(
        "--preprocessing",
        type=int,
        default=0,
        help="Index from : [implementations.z_normalize,implementations.quantile_normalize,implementations.min_max_normalize]",
    )
    parser.add_argument(
        "--train_val_ratio",
        type=float,
        default=0.95,
        help="The training/validation ratio to use for the given dataset",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--n_epochs", type=int, default=2, help="Number of train epochs"
    )
    parser.add_argument(
        "--k_fold", type=int, default=0, help="Number of folds for K-fold validation "
    )
    config = parser.parse_args()

    # Reproducibility:
    np.random.seed(config.seed)
    random.seed(config.seed)

    main(config)
