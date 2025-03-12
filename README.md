# Kernel methods

Lucas Versini

This repository contains my work on a [project](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2024-2025/) for the [Machine learning with kernel methods](https://mva-kernel-methods.github.io/course-page/) class, for which I was ranked 4th out of 37.

The goal of this project is to implement kernel methods for a classification task on DNA sequences. Machine Learning libraries such as `scikit-learn`, `libsvm`, etc. were forbidden: everything had to be implemented from scratch.

**As explained in the Data section below, downloading the files from [here](https://drive.google.com/drive/folders/1uFX_MT0YVCdv1JQ4DPwzT0souCI_9C4f?usp=sharing) will make execution faster. These files contain data for the mismatch kernel which were simply obtained by looking at the list of k-mers present in the datasets. If they are not downloaded, the scripts will compute them, which may take some time.**

## Installation

To install all required dependencies, you can run:
```bash
pip install -r requirements.txt
```

To create the submission, you can run:
```bash
python start.py
```

## Files
Here is a brief description of each file:
- `classifiers.py`: implementation of Ridge regression, Support Vector Machine and Logistic regression.
- `kernels.py`: implementation of Linear kernel, Polynomial kernel, Gaussian kernel, Spectrum kernel, Mismatch kernel, and sum of kernels.
- `utils.py`: implementation of a few utility functions used for the Mismatch kernel.
- `start.py`: to create the submission.

Moreover, the report can be found in `Versini_Lucas_report.pdf`.

## Data

By default, the scripts expect the data to be organized in a folder named `data`, which contains all the files `Xte{i}.csv`, `Xte{i}_mat100.csv`, `Xtr{i}.csv`, `Xtr{i}_mat100.csv` and `Ytr{i}.csv` for `i` equal to 0, 1, 2.

## Pre-computed data

For the mismatch kernel, we precomputed some data (the set of k-mers with Hamming distance at most m from a given k-mer, etc.), which can be found [here](https://drive.google.com/drive/folders/1uFX_MT0YVCdv1JQ4DPwzT0souCI_9C4f?usp=sharing).

These files should be placed in a folder `mismatch` in the same folder as the python scripts (otherwise, modify the path at the beginning of `utils.py`).

If these files are not downloaded, then the different scripts will compute them on the fly, which may significantly increase runtime.
