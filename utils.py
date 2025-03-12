import numpy as np
import pickle
import pandas as pd
import os

"""
This file contains functions to compute, or precompute, the mismatches for the mismatch kernel.
The first execution may be slow, but the next ones will be faster.
"""

alphabet = ['A', 'C', 'G', 'T']

path_to_mismatch = './mismatch'
if not os.path.exists(path_to_mismatch):
    os.makedirs(path_to_mismatch)

def build_kmer_set(X, k):
    """ Build the set of all kmers of length k in the dataset X """
    kmer_set = {kmer: i for i, kmer in enumerate(set(x[j : j + k] for x in X for j in range(len(x) - k + 1)))}
    return kmer_set

def get_mismatches_kmer(kmer, m):
    """ Get all the kmers at Hamming distance at most m from the kmer """
    def mutate(kmer_list, m, index):
        if m == 0:
            return {"".join(kmer_list)}
        
        neighbours_set = set()
        for i in range(index, len(kmer)): # Mutate starting from `index`
            original_char = kmer_list[i]
            for l in alphabet:
                kmer_list[i] = l
                neighbours_set.update(mutate(kmer_list, m - 1, i + 1))
            kmer_list[i] = original_char
        return neighbours_set

    return list(mutate(list(kmer), m, 0))

def get_mismatches_dataset(kmer_set, m):
    """ Get the mismatches for all the kmers in the dataset """
    neighbours = {kmer: [kmer_ for kmer_ in get_mismatches_kmer(kmer, m) if kmer_ in kmer_set]
                  for kmer in kmer_set}
    return neighbours

def load_mismatches(dataset, k, m):
    """ Load the mismatches from the disk """
    path = os.path.join(path_to_mismatch, f'mismatch_{dataset}_{k}_{m}.pkl')
    with open(path, 'rb') as f:
        mismatches, kmer_set = pickle.load(f)
    return mismatches, kmer_set

def compute_mismatch(dataset, k, m):
    """ Compute the mismatches for the dataset """
    X_train = pd.read_csv(f'data/Xtr{dataset}.csv', sep = ',', index_col = 0).values
    X_test = pd.read_csv(f'data/Xte{dataset}.csv', sep = ',', index_col = 0).values
    X = np.concatenate((X_train, X_test), axis = 0)[:, 0]

    kmer_set = build_kmer_set(X, k)
    mismatches = get_mismatches_dataset(kmer_set, m)

    path = os.path.join(path_to_mismatch, f'mismatch_{dataset}_{k}_{m}.pkl')
    with open(path, 'wb') as f:
        pickle.dump((mismatches, kmer_set), f)

    return mismatches, kmer_set

def load_compute_mismatches(dataset, k, m):
    """ Load the mismatches if they exist, otherwise compute them """
    try:
        mismatches, kmer_set = load_mismatches(dataset, k, m)
    except:
        print(f"Computing mismatches for dataset {dataset}, k = {k}, m = {m}")
        mismatches, kmer_set = compute_mismatch(dataset, k, m)
    return mismatches, kmer_set