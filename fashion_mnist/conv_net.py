# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:44:38 2019

@author: jason
"""

import numpy as np
from .conv_layer import ConvLayer
from .max_pool_layer import MaxPoolLayer
from .dense_layer import DenseLayer
from .utils import one_hot_encode
from .activation import activation_relu, activation_none


class ConvNet():

    def __init__(self,
                 learning_rate=0.0001,
                 batch_size=256,
                 num_classes=10,
                 num_epoch=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_epoch = num_epoch
        self.global_step = 0
        self.layers = []

    def _add_layer(self, layer):
        self.layers.append(layer)

    def _build_graph(self):
        self._add_layer(ConvLayer(
                input_shape=[self.batch_size, 28, 28, 1],
                filter_shape=[5, 5, 1, 32],
                strides=[1, 1]))  # output [batch_size, 28, 28, 32]
        self._add_layer(MaxPoolLayer(
                input_shape=[self.batch_size, 28, 28, 32],
                pool_shape=[2, 2],
                strides=[2, 2]))  # output [batch_size, 14, 14, 32]
        self._add_layer(ConvLayer(
                input_shape=[self.batch_size, 14, 14, 32],
                filter_shape=[5, 5, 32, 64],
                strides=[1, 1]))  # output [batch_size, 14, 14, 64]
        self._add_layer(MaxPoolLayer(
                input_shape=[self.batch_size, 14, 14, 64],
                pool_shape=[2, 2],
                strides=[2, 2]))  # output [batch_size, 7, 7, 64]
        self._add_layer(DenseLayer(
                input_shape=[self.batch_size, 7, 7, 64],
                output_shape=[self.batch_size, 10],
                activation=activation_none))
#        self._add_layer(DenseLayer(
#                input_shape=[self.batch_size, 7, 7, 64],
#                output_shape=[self.batch_size, 512],
#                activation=activation_relu))
#        self._add_layer(DenseLayer(
#                input_shape=[self.batch_size, 512],
#                output_shape=[self.batch_size, 10],
#                activation=activation_none))

    def _compute_shape(self, batch_size=None):
        for layer in self.layers:
            if batch_size:
                layer.compute_shape(batch_size)
            else:
                layer.compute_shape(self.batch_size)

    def _forward(self, input_):
        for layer in self.layers:
            input_ = layer.forward(input_)
        return input_

    def _compute_cross_entropy(self, output, labels):
        output_ = np.exp(output - np.amax(output, axis=1, keepdims=True))
        output_ = output_ / np.sum(output_, axis=1, keepdims=True)
        output_ += 1e-7
#        cross_entropy = -np.log(output_[labels == 1])
        cross_entropy_ = -np.sum(np.multiply(np.log(output_), labels), axis=1)
#        assert np.allclose(cross_entropy, cross_entropy_)
        # b: batch_size
        # ; means column vector
        # dl / dA = [softmax(A)_1: - Y_1:;
        #            ...;
        #            softmax(A)_b: - Y_b:]
        d_l_d_A = (output_ - labels)  # [batch_size, 10]
        return cross_entropy_, d_l_d_A

    def _backprop(self, d_l_d_A):
        for i, layer in enumerate(reversed(self.layers)):
            d_l_d_A = layer.backprop(d_l_d_A, self.learning_rate)
        return None

    def _gen_batch(self, X, y, batch_size):
        n_samples = X.shape[0]
        index = np.arange(n_samples)
        np.random.shuffle(index)
        current = 0
        while current + batch_size < n_samples:
            start, stop = current, current + batch_size
            yield X[index[start: stop]], y[index[start: stop]]
            current = stop
#        yield X[index[current:]], y[index[current:]]

    def fit(self, X, y):
        self._build_graph()
        self._compute_shape()
        X = X.reshape(-1, 28, 28, 1)
        y = one_hot_encode(y)
        for i in range(self.num_epoch):
            for j, (input_, labels) in enumerate(self._gen_batch(X, y, self.batch_size)):
                output = self._forward(input_)
                loss, d_l_d_A = self._compute_cross_entropy(output, labels)
                if j % 1 == 0:
                    loss = np.mean(loss)
                    acc = np.mean(np.equal(np.argmax(labels, axis=1), np.argmax(output, axis=1)))
                    print(f'{i} epoch {j} step batch_loss: {loss} batch_acc: {acc}')
                self._backprop(d_l_d_A)
                self.global_step += 1

    def predict(self, X):
        X = X.reshape(-1, 28, 28, 1)
        self._compute_shape(X.shape[0])
        return np.argmax(self._forward(X), axis=1)
