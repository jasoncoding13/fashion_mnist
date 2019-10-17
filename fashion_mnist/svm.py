#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:17:50 2019

@author: jason
"""

import numpy as np
from .utils import row_norms, print_log


class SVM():
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=None, coef0=0.0,
                 tol=1, max_iter=100000, skip_steps=10000):
        self.C = C
        if kernel == 'linear':
            self._compute_km = compute_linear_km
        elif kernel == 'poly':
            self._compute_km = compute_poly_km
        elif kernel == 'rbf':
            self._compute_km = compute_rbf_km
        elif kernel == 'sigmoid':
            self._compute_km = compute_sigmoid_km
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.skip_steps = skip_steps
        print_log(('{}'*4).format(C, kernel, degree, gamma))

    # initialize alphas
    def _initialize_alphas(self, n_samples):
        self.alphas = np.random.uniform(0, self.C, size=n_samples)
        while self.alphas.dot(self._y) > 0:
            self.alphas[self._y == 1] *= 0.999
        while self.alphas.dot(self._y) < 0:
            self.alphas[self._y == -1] *= 0.999
        diff = self.alphas.dot(self._y)
        for i in np.nonzero(self._y == 1)[0]:
            if self.alphas[i] > diff:
                self.alphas[i] -= diff
                break
            else:
                self.alphas[i] = 0
                diff = self.alphas.dot(self._y)

    def _compute_losses(self, kernel_matrix, E):
        alpha_y = (self.alphas * self._y)
        alpha_y_km_alpha_y = alpha_y[np.newaxis, :].dot(kernel_matrix).dot(alpha_y[:, np.newaxis])[0][0]
        dual_loss = int(np.sum(self.alphas) - 1 / 2 * alpha_y_km_alpha_y)
        primal_loss = int(1 / 2 * alpha_y_km_alpha_y + self.C * np.sum(np.maximum(-self. _y * E, 0)))
        return dual_loss, primal_loss

    def _gen_ind(self, E, n_samples):
#        E_ = E.copy()
        ind_nonKKT = np.nonzero((self.alphas != 0) & (self.alphas != self.C) & (self._y * E != 0))[0]
        np.random.shuffle(ind_nonKKT)
        for i in range(min(len(ind_nonKKT), 200)):
#            _ = np.argmax(np.abs(E_[ind_nonKKT[i]] - E_))
#            yield ind_nonKKT[i], _
#            E_[_] = 0
            yield ind_nonKKT[i], np.random.choice(n_samples)

    def _compute_kappa(self, kernel_matrix, ind1, ind2):
        return (kernel_matrix[ind1, ind1] + kernel_matrix[ind2, ind2] - 2 * kernel_matrix[ind1, ind2])

    def _compute_alpha2(self, E, ind1, ind2, kappa):
        alpha2_update = self.alphas[ind2] + (self._y[ind2] * (E[ind1] - E[ind2]) / kappa)
        if self._y[ind1] * self._y[ind2] < 0:
            lower = max(0, self.alphas[ind2] - self.alphas[ind1])
            upper = min(self.C, self.C + self.alphas[ind2] - self.alphas[ind1])
        else:
            lower = max(0, self.alphas[ind1] + self.alphas[ind2] - self.C)
            upper = min(self.C, self.alphas[ind1] + self.alphas[ind2])
        if alpha2_update > upper:
            alpha2_update = upper
        elif alpha2_update < lower:
            alpha2_update = lower
        return alpha2_update

    def _compute_alpha1(self, ind1, ind2, alpha2_update):
        return self.alphas[ind1] + (self._y[ind1] * self._y[ind2] * (self.alphas[ind2] - alpha2_update))

    def fit(self, X, y):
        n_samples, _ = X.shape
        # assign gamma
        if self.gamma is 'auto':
            self.gamma = 1.0 / (X.shape[1])
        if self.gamma is 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var())
        # encode 0 > -1, 1> 1
        self._y = np.where(y == 1, 1, -1)
        # initilize alphas
        self._initialize_alphas(n_samples)
        # initilize b
        self.b = 0
        # compute kernel matrix
        kernel_matrix = self._compute_km(X, X, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        # compute E
        E = (self.alphas * self._y).dot(kernel_matrix) + self.b - self._y
        # generate ind1, ind2
        g_ind = self._gen_ind(E, n_samples)
        ind1, ind2 = next(g_ind)
        # SMO
        trigger = True
        self.dual_losses, self.primal_losses = [[loss] for loss in self._compute_losses(kernel_matrix, E)]
        self.n_sv = [n_samples]
        iter_ = 0
        while trigger:
            # compute kappa
            kappa = self._compute_kappa(kernel_matrix, ind1, ind2)
            if kappa > 0:
                # compute alpha2
                alpha2_update = self._compute_alpha2(E, ind1, ind2, kappa)
                # compute alpha1
                alpha1_update = self._compute_alpha1(ind1, ind2, alpha2_update)
                # update b
                b1 = (self.b
                      - E[ind1]
                      - self._y[ind1] * (alpha1_update - self.alphas[ind1]) * kernel_matrix[ind1, ind1]
                      - self._y[ind2] * (alpha2_update - self.alphas[ind2]) * kernel_matrix[ind1, ind2])
                b2 = (self.b
                      - E[ind2]
                      - self._y[ind1] * (alpha1_update - self.alphas[ind1]) * kernel_matrix[ind1, ind2]
                      - self._y[ind2] * (alpha2_update - self.alphas[ind2]) * kernel_matrix[ind2, ind2])
                # update E
                E = (E
                     - (self.alphas[ind1] * self._y[ind1] * kernel_matrix[ind1, :])
                     - (self.alphas[ind2] * self._y[ind2] * kernel_matrix[ind2, :])
                     + (alpha1_update * self._y[ind1] * kernel_matrix[ind1, :])
                     + (alpha2_update * self._y[ind2] * kernel_matrix[ind2, :])) - self.b
                self.b = (b1+b2)/2
                E += self.b
                # update alpha1, alpha2
                self.alphas[ind1] = alpha1_update
                self.alphas[ind2] = alpha2_update
            # generate ind1
            try:
                ind1, ind2 = next(g_ind)
            except (StopIteration, TypeError):
                try:
                    g_ind = self._gen_ind(E, n_samples)
                    ind1, ind2 = next(g_ind)
                except (StopIteration, TypeError):
                    self.n_sv.append(np.sum((self.alphas > 0)))
                    dual_loss, primal_loss = self._compute_losses(kernel_matrix, E)
                    self.dual_losses.append(dual_loss)
                    self.primal_losses.append(primal_loss)
                    print('At most 1 samples satisfy KKT, iter_: {:>5}, n_sv: {:>5}, losses:{:>15}, {:>15}'.format(
                            iter_, self.n_sv[-1], self.dual_losses[-1], self.primal_losses[-1]))
                    trigger = False
            # check convergence
            if iter_ % self.skip_steps == 0:
                print('iter_: {:>5}, n_sv: {:>5}, losses: {:>13}, {:>13}'.format(
                        iter_, self.n_sv[-1], self.dual_losses[-1], self.primal_losses[-1]))
                self.n_sv.append(np.sum(self.alphas > 0))
                dual_loss, primal_loss = self._compute_losses(kernel_matrix, E)
                self.dual_losses.append(dual_loss)
                self.primal_losses.append(primal_loss)
                if iter_ == self.max_iter:
                    print('Converged, iter_: {:>5}, n_sv: {:>5}, losses:{:>13}, {:>13}'.format(
                            iter_, self.n_sv[-1], self.dual_losses[-1], self.primal_losses[-1]))
                    trigger = False
            iter_ += 1
        ind_sv = self.alphas > 0
        self._y = self._y[ind_sv]
        self._X = X[ind_sv, :]
        self.alphas = self.alphas[ind_sv]
        kernel_matrix_sv = self._compute_km(self._X, self._X, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        self.b = np.mean(1 / self._y - (self.alphas * self._y).dot(kernel_matrix_sv))
        return self

    def predict(self, X):
        kernel_matrix_pred = self._compute_km(self._X, X, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        y_pred = (self.alphas * self._y).dot(kernel_matrix_pred) + self.b
        y_pred = np.where(y_pred < 0, 0, 1)
        return y_pred

    def predict_proba(self, X):
        y_pred_proba = np.empty((X.shape[0], 2))
        kernel_matrix_pred = self._compute_km(self._X, X, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        exp_ = np.exp((self.alphas * self._y).dot(kernel_matrix_pred) + self.b)
        y_pred_proba[:, 0] = 1 / (1 + exp_)
        y_pred_proba[:, 1] = 1 - y_pred_proba[:, 0]
        return y_pred_proba


def compute_linear_km(X, Y, **kwargs):
    return X.dot(Y.T)


def compute_poly_km(X, Y, **kwargs):
    degree = kwargs.get('degree', 3)
    gamma = kwargs.get('gamma', None)
    coef0 = kwargs.get('coef0', 1)
    if gamma is None:
        gamma = 1.0 / (X.shpae[1] * X.val())
    km = X.dot(Y.T)
    km *= gamma
    km += coef0
    km **= degree
    return km


def compute_rbf_km(X, Y, **kwargs):
    gamma = kwargs.get('gamma', None)
    if gamma is None:
        gamma = 1.0 / (X.shape[1])
    XX = row_norms(X, squared=True)[:, np.newaxis]  # [n_rows, 1]
    YY = row_norms(Y, squared=True)[np.newaxis, :]  # [1, n_rows]
    km = - 2 * X.dot(Y.T)
    km += XX
    km += YY
    np.maximum(km, 0, out=km)
    km = km.astype(np.float64)
    km *= -gamma
    np.exp(km, out=km)
    return km


def compute_sigmoid_km(X, Y, **kwargs):
    gamma = kwargs.get('gamma', None)
    if gamma is None:
        gamma = 1.0 / (X.shpae[1] * X.val())
    coef0 = kwargs.get('coef0', -1)
    km = X.dot(Y.T)
    km *= gamma
    km += coef0
    np.tanh(km, out=km)
    return km
