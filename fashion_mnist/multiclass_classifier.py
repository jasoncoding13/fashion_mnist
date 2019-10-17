# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:41:52 2019

@author: Jason
@e-mail: jasoncoding13@gmail.com
"""

import numpy as np


def binarize_label(label, zero_label):
    return np.where(label == zero_label, 0, 1)


class OneVsRestClassifier():

    def __init__(self, base_clf, **params):
        self.base_clf = base_clf
        self.clfs = {}
        self.params = params

    def fit(self, X_train, y_train):
        y_unique = np.unique(y_train)
        self.n_classes = y_unique.shape[0]
        for _, y in enumerate(y_unique):
            print(f'OVR: {y}')
            _y_train_binarized = binarize_label(y_train, zero_label=y)
            clf = self.base_clf(**self.params)
            clf.fit(X_train, _y_train_binarized)
            self.clfs[y] = clf

    def predict(self, X_test):
        matrix_proba = np.zeros(
                [X_test.shape[0], self.n_classes], dtype=np.float)
        for y, clf in self.clfs.items():
            matrix_proba[:, y] = clf.predict_proba(X_test)[:, 0]
        return np.argmax(matrix_proba, axis=1)


class OneVsOneClassifier():

    def __init__(self, base_clf, **params):
        self.base_clf = base_clf
        self.clfs = {}
        self.params = params

    def fit(self, X_train, y_train):
        y_unique = np.unique(y_train)
        self.n_classes = y_unique.shape[0]

        for i, y1 in enumerate(y_unique[:-1]):
            for y2 in y_unique[i+1:]:
                print(f'OVO: {y1}-{y2}')
                _mask = (y_train == y1) | (y_train == y2)
                _y_train_binarized = binarize_label(
                        y_train[_mask], zero_label=y1)
                clf = self.base_clf(**self.params)
                clf.fit(X_train[_mask], _y_train_binarized)
                self.clfs[f'{y1}_{y2}'] = clf

    def predict(self, X_test):
        matrix_vote = np.zeros(
                [X_test.shape[0], self.n_classes], dtype=np.int)
        for vs, clf in self.clfs.items():
            y1, y2 = list(map(int, vs.split('_')))
            matrix_vote[clf.predict(X_test) == 0, y1] += 1
            matrix_vote[clf.predict(X_test) == 1, y2] += 1
        return np.argmax(matrix_vote, axis=1)
