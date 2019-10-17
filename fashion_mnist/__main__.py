# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:57:16 2019

@author: Jason
@e-mail: jasoncoding13@gmail.com
"""


import numpy as np
import sys
from sklearn.svm import SVC
from .conv_net import ConvNet
from .multiclass_classifier import OneVsOneClassifier
from .svm import SVM
from .utils import load_data


def main_svm():
    X_train, y_train = load_data('./data', kind='train')
    X_test, y_test = load_data('./data', kind='t10k')
    accs = []
    for clf in [OneVsOneClassifier(SVM, C=10, gamma='scale', max_iter=30000),
                SVC(C=10, gamma='scale', max_iter=30000, decision_function_shape='ovo')]:
        clf.fit(X_train, y_train)
        accs.append(np.mean(clf.predict(X_test)) == y_test)
    print('test accuracy from scratch: {}, from sklearn: {}'.format(*accs))


def main_cnn():
    X_train, y_train = load_data('./data', kind='train')
    X_test, y_test = load_data('./data', kind='t10k')
    clf = ConvNet()
    clf.fit(X_train, y_train)
    accs = []
    for i in range(100):
        y_pred = clf.predict(X_test[i*100:(i+1)*100, :])
        accs.append(np.mean(y_pred == y_test[i*100:(i+1)*100]))
    print('test accuracy from cnn:{}'.format(np.mean(accs)))


if __name__ == '__main__':
    if sys.argv[1] == 'svm':
        main_svm()
    elif sys.argv[1] == 'cnn':
        main_cnn()
    else:
        print('please use `python -m fashion_mnist fashion_mnist svm` or \n',
                         '`python -m fashion_mnist.fashion_mnist cnn')