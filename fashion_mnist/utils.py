# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:58:13 2019

@author: Jason
@e-mail: jasoncoding13@gmail.com
"""

import gzip
import numpy as np
import os
import sys
import time
from scipy.stats import truncnorm, uniform


def load_data(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images/255.0, labels


def gen_valid_index(index, k):
    """
    A generator to shuffle and split index in k folds.
    For example, index = [0,1,2], k = 3, the first index array returned is [0],
    the sencond [1] and the third [2]. Note that it can be used to split index
    of the same class for stratified k folds cross validation.

    Parameters:
    ----------
    index: ndarray, an array of index used to be splited into folds

    k: int, the number of folds

    Returns:
    index[start: stop]
    """
    n_samples = index.shape[0]
    np.random.shuffle(index)
    # an array of shape `(k, )` filled with `n_samples // k`
    fold_sizes = np.full(k, n_samples//k, dtype=np.int)
    # handle case in `n_samples % k > 0`
    # make `sum(fold_sizes) ` equal to n_samples
    # For example, n_samples = 13, k = 5, fold_sizes = [3, 3, 3, 2, 2]
    fold_sizes[:n_samples % k] += 1
    # 2 pointers for start and stop
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield index[start: stop]
        current = stop


def gen_cv_mask(y_train, n_folds=5):
    """
    A generator to generate masks for stratified k fold cross validation.
    For example, y_train = [0, 1, 1, 0, 1, 1], n_folds = 2.
    Ignoring the effect of shuffle,
    for class of 0, it uses `gen_valid_index` to obtain [0], [3].
    For class of 1, it obtains [1, 2], [3, 4].
    The index for first fold is [0, 1, 2].
    The masks for first fold is [True, True, True, False, False, False].
    The masks for second fold is [False, False, False, True, True, True].
    """
    y_unique = np.unique(y_train)
    mask_fold = np.zeros(y_train.shape[0], dtype=np.int)
    for i, y in enumerate(y_unique):
        index_group = np.nonzero(y_train == y)[0]
        g_valid_index = gen_valid_index(index_group, n_folds)
        for k in range(n_folds):
            _index = next(g_valid_index)
            mask_fold[_index] = k
            assert np.all(y_train[_index] == y)
    for k in range(n_folds):
        yield mask_fold == k


# feature normalizer
class Normalizer():
    def __init__(self, method='minmax'):
        self.method = method

    def fit_transform(self, X_train):
        if self.method == 'minmax':
            self.loc = 0
            self.scale = 1
        elif self.method == 'zscore':
            self.loc = np.mean(X_train)
            self.scale = np.std(X_train)
        print(f'normalizer: {self.loc}, {self.scale}')
        return (X_train - self.loc) / self.scale

    def transform(self, X_test):
        return (X_test - self.loc) / self.scale

def print_log(string):
    sys.stdout.write(string+'\n')
    sys.stdout.flush()

# utils for CNN
def row_norms(X, squared=False):
    norms = np.einsum('ij,ij->i', X, X)
    if not squared:
        np.sqrt(norms, out=norms)
    return norms


def get_variables(shape, init='uniform', loc=0, scale=1):
    np.random.seed(13)
    if init == 'ones':
        return np.ones(shape)
    elif init == 'norm':
        return truncnorm.rvs(-1.96, 1.96, loc=loc, scale=scale, size=np.prod(shape)).reshape(shape)
    elif init == 'uniform':
        return uniform.rvs(loc=loc, scale=scale, size=np.prod(shape)).reshape(shape)


def one_hot_encode(y):
    y_unique = np.unique(y)
    y_ret = np.zeros([y.shape[0], y_unique.shape[0]])
    for i, e in enumerate(y_unique):
        y_ret[y == e, i] = 1
    return y_ret


def test_layer(baselayer, layer, type_=None):
    b = baselayer()
    l = layer()
    input_shape = b.input_shape
    input_shape[0] = 128
    b.compute_shape(input_shape[0])
    l.compute_shape(input_shape[0])
    input_ = np.random.rand(*input_shape)
    start = time.time()
    output_b = b.forward(input_)
    stop = time.time()
    print('baselayer forward:', stop - start)
    start = time.time()
    output_l = l.forward(input_)
    stop = time.time()
    print('layer forward:', stop - start)
    assert np.allclose(output_b, output_l)
    d_l_d_A = 1 - output_b
    start = time.time()
    d_l_d_input_b = b.backprop(d_l_d_A)
    stop = time.time()
    print('baselayer backprop:', stop-start)
    start = time.time()
    d_l_d_input_l = l.backprop(d_l_d_A)
    stop = time.time()
    print('layer backprop:', stop-start)
    assert np.allclose(d_l_d_input_b, d_l_d_input_l)
    if type_ == 'conv':
        assert np.allclose(b.filters, l.filters)
        assert np.allclose(b.biases, l.biases)
    elif type_ == 'dense':
        assert np.allclose(b.weights, l.weights)
        assert np.allclose(b.weights, l.biases)
