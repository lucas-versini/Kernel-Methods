import pandas as pd
import numpy as np

np.random.seed(0)

from kernels import get_kernel
from classifiers import get_classifier
from utils import load_compute_mismatches

perform_cross_validation = False

# Parameters (for each dataset)
params = {}
params['kernel_type'] = 'sum'
list_k_m = [(5, 1), (8, 1), (10, 1), (12, 2), (13, 2), (15, 3)]
params['kernels'] = [('mismatch', {'k': k, 'm': m, 'normalize': True}) for k, m in list_k_m]
params['weights'] = [0.18528693, 0.17195168, 0.22389528, 0.08329321, 0.09638722, 0.23918568]
params['classifier_type'] = 'svm'
params['lbd'] = 1.0
params['C'] = 5.
params_list = [params.copy() for _ in range(3)] # Same parameters for each dataset

# params = {}
# params['kernel_type'] = 'gaussian'
# params['classifier_type'] = 'svm'
# params['kernels'] = []
# params['C'] = 3.
# params['sigma'] = 0.05
# params_list = [params.copy() for _ in range(3)]
# params_list[1]['C'] = 9.
# params_list[2]['C'] = 0.9

# Load data
X_train_list, X_mat100_train_list, Y_train_list = [], [], []
X_test_list, X_mat100_test_list = [], []

for i in range(3):
    X_train = pd.read_csv(f"data/Xtr{i}.csv", sep = ",", index_col = 0).values[:, 0]
    X_mat100_train = pd.read_csv(f"data/Xtr{i}_mat100.csv", sep = " ", header = None).values
    Y_train = pd.read_csv(f"data/Ytr{i}.csv", sep = ",", index_col = 0).values[:, 0]
    Y_train = 2 * Y_train - 1

    X_test = pd.read_csv(f"data/Xte{i}.csv", sep = ",", index_col = 0).values[:, 0]
    X_mat100_test = pd.read_csv(f"data/Xte{i}_mat100.csv", sep = " ", header = None).values

    X_train_list.append(X_train)
    X_mat100_train_list.append(X_mat100_train)
    Y_train_list.append(Y_train)
    X_test_list.append(X_test)
    X_mat100_test_list.append(X_mat100_test)

def k_fold_cross_validation(X_mat, X_str, Y, k, classifier):
    """ Perform a k-fold cross-validation
    
    Parameters:
    X_mat: np.ndarray, shape (n_samples, n_features)
            The input data (numerical features)
    X_str: np.ndarray, shape (n_samples, n_features)
            The input data (string features)
    Y: np.ndarray, shape (n_samples,)
            The target data
    k: int
            The number of folds
    classifier: object
            The classifier object
    """
    np.random.seed(0)
    indices = np.random.permutation(X_mat.shape[0])
    X_mat, X_str, Y = X_mat[indices], X_str[indices], Y[indices]
    X_mat_folds, X_str_folds, Y_folds = np.array_split(X_mat, k), np.array_split(X_str, k), np.array_split(Y, k)
    scores = []
    for i in range(k):
        X_mat_train = np.concatenate([X_mat_folds[j] for j in range(k) if j != i])
        X_str_train = np.concatenate([X_str_folds[j] for j in range(k) if j != i])
        Y_train = np.concatenate([Y_folds[j] for j in range(k) if j != i])
        X_mat_test = X_mat_folds[i]
        X_str_test = X_str_folds[i]
        Y_test = Y_folds[i]

        classifier.fit(X_mat_train, X_str_train, Y_train.astype(float))
        Y_pred = classifier.predict_class(X_mat_test, X_str_test)

        score = np.mean(Y_pred == Y_test)
        scores.append(score)
        print(f"Fold {i + 1}/{k}: {score}")
    return np.mean(scores)

def make_prediction(classifier, X_mat_train, X_train, Y_train, X_mat_test, X_test):
    """ Make a prediction"""
    classifier.fit(X_mat_train, X_train, Y_train)
    Y_pred_class = classifier.predict_class(X_mat_test, X_test)

    Y_pred_class = (Y_pred_class > 0).astype(int)

    return Y_pred_class

def make_submission(params_list, X_mat100_train_list, X_train_list, Y_train_list, X_mat100_test_list, X_test_list):
    """ Make a submission """
    list_Y_pred_class = []

    for dataset, (params, X_mat100_train, X_train, Y_train, X_mat100_test, X_test) in enumerate(zip(params_list, X_mat100_train_list, X_train_list, Y_train_list, X_mat100_test_list, X_test_list)):
        for kernel_type, kernel_params in params['kernels']:
            if kernel_type == 'mismatch':
                mismatches, kmer_set = load_compute_mismatches(dataset, kernel_params['k'], kernel_params['m'])
                kernel_params['kmer_set'] = kmer_set
                kernel_params['mismatches'] = mismatches

        kernel = get_kernel(params['kernel_type'], params)
        classifier = get_classifier(params['classifier_type'], kernel, params)
        Y_pred_class = make_prediction(classifier, X_mat100_train, X_train, Y_train.astype(float), X_mat100_test, X_test)
        list_Y_pred_class.append(Y_pred_class)

    # Save concatenated predictions
    Y_pred_class = np.concatenate(list_Y_pred_class)
    if Y_pred_class.shape != (3000, 1):
        Y_pred_class = Y_pred_class.reshape(-1, 1)
    df = pd.DataFrame(Y_pred_class, columns = ['Bound'])
    df.index.name = 'Id'
    df.to_csv("Yte.csv")

# Cross-validation
if perform_cross_validation:
    list_score = []
    for i, (params, X_mat100_train, X_train, Y_train) in enumerate(zip(params_list, X_mat100_train_list, X_train_list, Y_train_list)):
        for kernel_type, kernel_params in params['kernels']:
            if kernel_type == 'mismatch':
                mismatches, kmer_set = load_compute_mismatches(i, kernel_params['k'], kernel_params['m'])
                kernel_params['kmer_set'] = kmer_set
                kernel_params['mismatches'] = mismatches
        kernel = get_kernel(params['kernel_type'], params)
        classifier = get_classifier(params['classifier_type'], kernel, params)
        k = 5
        score = k_fold_cross_validation(X_mat100_train, X_train, Y_train, k, classifier)
        print(f"Cross-validation score {i}: {score}")
        list_score.append(score)
    print(f"Mean cross-validation score: {np.mean(list_score)}")

# Make submission
make_submission(params_list, X_mat100_train_list, X_train_list, Y_train_list, X_mat100_test_list, X_test_list)

print("Submission created in Yte.csv")
