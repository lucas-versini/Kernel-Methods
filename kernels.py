import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import Counter

"""
This file contains the implementation of:
- Linear Kernel
- Polynomial Kernel
- Gaussian Kernel
- Spectrum Kernel
- Mismatch Kernel
- Sum Kernel

Note that whenever we want to compute the gram matrix, we take as input both the numerical and string representations of the sequences.
This way, the rest of the code works for string kernels and numerical kernels.
"""

def get_kernel(kernel_type, params):
    if kernel_type == 'linear':
        return LinearKernel()
    elif kernel_type == 'polynomial':
        return PolynomialKernel(params['degree'], params['c'])
    elif kernel_type == 'gaussian':
        return GaussianKernel(params['sigma'])
    elif kernel_type == 'spectrum':
        return SpectrumKernel(params['k'])
    elif kernel_type == 'mismatch':
        k, m = params['k'], params['m']
        kmer_set, mismatches = params['kmer_set'], params['mismatches']
        normalize = params['normalize']
        return MismatchKernel(k, m, kmer_set, mismatches, normalize)
    elif kernel_type == 'sum':
        kernels = [get_kernel(k, p) for (k, p) in params['kernels']]
        weights = params['weights']
        types = [p for p, _ in params['kernels']]
        return SumKernel(kernels, weights, types)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

class LinearKernel:
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        return x1 @ x2.T
    
    def gram_matrix(self, X1, X2):
        return X1[0] @ X2[0].T

class PolynomialKernel:
    def __init__(self, degree = 2, c = 1.0):
        self.degree = degree # degree of the polynomial kernel
        self.c = c # constant term in the polynomial kernel

    def __call__(self, x1, x2):
        return (x1 @ x2.T + self.c)**self.degree
    
    def gram_matrix(self, X1, X2):
        return (X1[0] @ X2[0].T + self.c)**self.degree
    
class GaussianKernel:
    def __init__(self, sigma = 0.1):
        self.sigma = sigma # standard deviation of the Gaussian kernel

    def __call__(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.sigma**2))
    
    def gram_matrix(self, X1, X2):
        return np.exp(-np.sum((X1[0][:, None] - X2[0][None, :])**2, axis = 2) / (2 * self.sigma**2))

class SpectrumKernel:
    def __init__(self, k = 5):
        self.k = k # length of the k-mers

    def __call__(self, x1, x2):
        subsequences_x1, count_x1 = np.unique([x1[i:i + self.k] for i in range(len(x1) - self.k + 1)], return_counts = True)
        return np.sum(np.char.count(x2, subsequences_x1) * count_x1)

    def gram_matrix(self, X1, X2):
        X1, X2 = X1[1], X2[1] # Use the string representation of the sequences
        gram = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            subsequences_x1, count_x1 = np.unique([x1[i:i + self.k] for i in range(len(x1) - self.k + 1)], return_counts = True)
            for j, x2 in enumerate(X2):
                gram[i, j] = np.sum(np.char.count(x2, subsequences_x1) * count_x1)
        return gram.astype(float)

class MismatchKernel():
    def __init__(self, k, m, kmer_set, mismatches, normalize = False):
        self.k = k # length of the k-mers
        self.m = m # number of mismatches allowed
        self.kmer_set = kmer_set # set of all possible k-mers
        self.mismatches = mismatches # dictionary of mismatches
        self.normalize = normalize # whether to normalize the kernel
    
    def feature_map(self, x):
        """ Compute the feature map of a sequence x """
        x_array = np.array(list(x))
        kmer_x = np.lib.stride_tricks.sliding_window_view(x_array, self.k)
        kmer_x = np.apply_along_axis(lambda k: ''.join(k), 1, kmer_x)

        neighbors = [neigh for kmer in kmer_x for neigh in self.mismatches.get(kmer, []) if neigh in self.kmer_set]

        x_emb = Counter(map(self.kmer_set.get, neighbors))

        return dict(x_emb)

    def feature_map_data(self, X):
        """ Compute the feature map of a list of sequences """
        return [self.feature_map(x) for x in X]
    
    def to_sparse(self, X_emb):
        """ Convert the feature map of a sequence to a sparse matrix
        to be able to compute the kernel matrix efficiently """
        data = np.concatenate([list(x.values()) for x in X_emb])
        row = np.concatenate([list(x.keys()) for x in X_emb])
        col = np.concatenate([[i] * len(x) for i, x in enumerate(X_emb)])

        return sparse.coo_matrix((data, (row, col)), shape = (len(self.kmer_set), len(X_emb)))

    def __call__(self, x, y):
        x_emb = self.feature_map(x)
        y_emb = self.feature_map(y)
        sp = sum(x_emb[idx_neigh] * y_emb[idx_neigh] for idx_neigh in x_emb if idx_neigh in y_emb)
        if self.normalize:
            sp /= np.sqrt(np.sum(np.array(list(x_emb.values()))**2))
            sp /= np.sqrt(np.sum(np.array(list(y_emb.values()))**2))
        return sp

    def gram_matrix(self, X1, X2):
        X1, X2 = X1[1], X2[1] # Use the string representation of the sequences

        X1_sm = self.to_sparse(self.feature_map_data(X1))
        X2_sm = self.to_sparse(self.feature_map_data(X2))

        G = X1_sm.T @ X2_sm
        G = G.todense().astype(float)
        
        if self.normalize:
            norms_X1 = np.sqrt(np.sum(X1_sm.power(2), axis = 0))
            norms_X2 = np.sqrt(np.sum(X2_sm.power(2), axis = 0))
            G /= np.array(norms_X1.T @ norms_X2)
            
        return G

class SumKernel():
    def __init__(self, kernels, weights, types):
        self.kernels = kernels # list of kernels
        self.weights = weights # list of weights
        self.types = types # list of types of the kernels

    def __call__(self, x, y):
        return sum(w * k(x, y) for k, w in zip(self.kernels, self.weights))

    def gram_matrix(self, X1, X2):
        return sum(w * k.gram_matrix(X1, X2) for k, w in zip(self.kernels, self.weights))