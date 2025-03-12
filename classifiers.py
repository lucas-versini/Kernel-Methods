import numpy as np
import cvxopt

"""
This file contains the implementation of:
- Ridge Classifier
- Support Vector Machine
- Logistic Regression
"""

def get_classifier(classifier_type, kernel, params):
    if classifier_type == 'ridge':
        return RidgeClassifier(kernel, params['lbd'])
    elif classifier_type == 'svm':
        return SVM(kernel, params['C'])
    elif classifier_type == 'logistic':
        return LogisticRegression(kernel, params['lbd'])
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class RidgeClassifier:
    def __init__(self, kernel, lbd = 1.0):
        """ Ridge Classifier
        
        Parameters:
        kernel: callable
            The kernel function to use
        lbd: float
            The regularization parameter
        """
        self.lbd = lbd
        self.kernel = kernel

    def fit(self, X_mat, X_str, y):
        """ Fit the model

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)
        y: np.ndarray, shape (n_samples,)
            The target data        
        """
        self.X_mat_train = X_mat
        self.X_str_train = X_str
        self.y_train = y
        self.K = self.kernel.gram_matrix((X_mat, X_str), (X_mat, X_str))
        if isinstance(self.K, np.matrix):
            self.K = np.array(self.K)
        self.alpha = np.linalg.inv(self.K + self.lbd * np.eye(len(X_mat))) @ y

    def predict(self, X_mat, X_str):
        """ Predict the target values

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)

        Returns:
        np.ndarray, shape (n_samples,)
            The predicted target values
        """
        K = self.kernel.gram_matrix((X_mat, X_str), (self.X_mat_train, self.X_str_train))
        if isinstance(K, np.matrix):
            K = np.array(K)
        return K @ self.alpha.squeeze()
    
    def predict_class(self, X_mat, X_str):
        """ Predict the class labels

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)

        Returns:
        np.ndarray, shape (n_samples,)
            The predicted class labels
        """
        return 2 * (self.predict(X_mat, X_str) > 0) - 1

class SVM():
    def __init__(self, kernel, C = 1.0, tol = 1e-6):
        """ Support Vector Machine
        
        Parameters:
        kernel: callable
            The kernel function to use
        C: float
            The regularization parameter
        tol: float
            The tolerance for the support vectors
        """
        self.kernel = kernel
        self.C = C
        self.tol = tol

    def fit(self, X_mat, X_str, y, verbose = False):
        """ Fit the model

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)
        y: np.ndarray, shape (n_samples,)
            The target data
        verbose: bool
            Whether to print the number of support vectors
        """
        y = y.astype(float)

        self.X_mat_train = X_mat
        self.X_str_train = X_str
        n = X_mat.shape[0]
        self.K_train = self.kernel.gram_matrix((X_mat, X_str), (X_mat, X_str))

        # Solve the quadratic optimization problem
        P = cvxopt.matrix(self.K_train)
        q = cvxopt.matrix(-y)
        G = cvxopt.matrix(np.concatenate((np.diag(y), -np.diag(y))))
        h = cvxopt.matrix(np.concatenate((self.C * np.ones(n), np.zeros(n))))

        solver = cvxopt.solvers.qp(P = P, q = q, G = G, h = h, options = {'show_progress': False})
        self.alpha = np.array(solver['x']).squeeze()

        # Find the support vectors
        indices = (np.abs(self.alpha) > self.tol)
        self.alpha = self.alpha[indices]
        self.support_vectors_mat = self.X_mat_train[indices]
        self.support_vectors_str = self.X_str_train[indices]

        if verbose:
            print(f"{len(self.support_vectors_mat)} support vectors out of {len(self.X_mat_train)} training samples")

    def predict(self, X_mat, X_str):
        """ Predict the target values

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)

        Returns:
        np.ndarray, shape (n_samples,)
            The predicted target values
        """
        K = self.kernel.gram_matrix((X_mat, X_str), (self.support_vectors_mat, self.support_vectors_str))
        y = K @ self.alpha
        return y

    def predict_class(self, X_mat, X_str, threshold = 0):
        """ Predict the class labels

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)
        threshold: float
            The threshold for the decision function

        Returns:
        np.ndarray, shape (n_samples,)
            The predicted class labels
        """
        return 2 * (self.predict(X_mat, X_str) > threshold) - 1

class LogisticRegression:
    def __init__(self, kernel, lbd = 1.0, tol = 1e-6):
        """ Logistic Regression
        
        Parameters:
        kernel: callable
            The kernel function to use
        lbd: float
            The regularization parameter
        tol: float
            The tolerance for the optimization
        """
        self.kernel = kernel
        self.lbd = lbd
        self.tol = tol

    def fit(self, X_mat, X_str, y, max_iter = 10):
        """ Fit the model

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)
        y: np.ndarray, shape (n_samples,)
            The target data
        max_iter: int
            The maximum number of iterations
        """
        n = X_mat.shape[0]
        self.X_mat_train, self.X_str_train = X_mat, X_str
        self.y_train = y
        self.K_train = self.kernel.gram_matrix((X_mat, X_str), (X_mat, X_str))

        self.alpha = np.zeros(n)

        for _ in range(max_iter):
            alpha_old = self.alpha.copy()

            if isinstance(self.K_train, np.matrix):
                self.K_train = np.array(self.K_train)
            m = (self.K_train @ self.alpha)
            m = m.squeeze()
            W = sigmoid(m) * sigmoid(-m)
            z = m + y / sigmoid(y * m)

            W_sqrt = np.sqrt(W)
            self.alpha = W_sqrt * np.linalg.solve(W_sqrt * self.K_train * W_sqrt + n * self.lbd * np.eye(n),
                                                  W_sqrt * z)

            if np.linalg.norm(self.alpha - alpha_old) < self.tol:
                break

    def predict(self, X_mat, X_str):
        """ Predict the target values

        Parameters:
        X_mat: np.ndarray, shape (n_samples, n_features)
                The input data (numerical features)
        X_str: np.ndarray, shape (n_samples, n_features)
                The input data (string features)
        
        Returns:
        np.ndarray, shape (n_samples,)
            The predicted target values
        """
        K = self.kernel.gram_matrix((X_mat, X_str), (self.X_mat_train, self.X_str_train))
        return K @ self.alpha

    def predict_class(self, X_mat, X_str, threshold = 0.):
        """ Predict the class labels

        Parameters:
        X: np.ndarray, shape (n_samples, n_features)
            The input data
        threshold: float
            The threshold for the decision function
        
        Returns:
        np.ndarray, shape (n_samples,)
            The predicted class labels
        """
        return 2 * (self.predict(X_mat, X_str) > threshold) - 1