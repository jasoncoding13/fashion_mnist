# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:12:54 2019

@author: jason
"""
import math
import numpy as np
from .utils import get_variables
from .activation import activation_relu, activation_none

class DenseLayerBase():

    def __init__(self,
                 input_shape=[-1, 7, 7, 64],
                 output_shape=[-1, 10],
                 activation=activation_none):
        self.input_shape = np.array(input_shape)
        self.output_shape = output_shape
        limit = math.sqrt(6 / (np.prod(self.input_shape[1:]) + self.output_shape[1]))
        self.weights = get_variables(np.r_[self.input_shape[1:],
                                           self.output_shape[1]], loc=-limit, scale=2*limit)
        self.biases = np.zeros(self.output_shape[1])
        self.activation = activation()

    def compute_shape(self, batch_size):
        self.input_shape[0] = batch_size
        self.output_shape[0] = batch_size

    def forward(self, input_):
        self.input_ = input_
        output = np.zeros(self.output_shape)
        for i in range(self.input_shape[0]):
            for j in range(self.output_shape[1]):
                # z = w^Tx + b for i-th sample j-th class
                # a = g(z)
                output[i, j] = self.activation.activate(
                        np.sum(np.multiply(input_[i, :, :, :], self.weights[:, :, :, j])) + self.biases[j])
        self.last_output = output
        return output

    def backprop(self, d_l_d_A, learning_rate=0.001):
        d_l_d_W = np.zeros_like(self.weights)
        d_l_d_b = np.zeros(self.output_shape[1])
        d_l_d_input = np.zeros(self.input_shape)
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                for k in range(self.input_shape[2]):
                    for l in range(self.input_shape[3]):
                        for m in range(self.output_shape[1]):
                            d_l_d_z = self.activation.derive(self.last_output[i, m])
                            d_l_d_W[j, k, l, m] += self.input_[i, j, k, l] * d_l_d_z * d_l_d_A[i, m]
                            d_l_d_b[m] += d_l_d_z * d_l_d_A[i, m]
                            d_l_d_input[i, j, k, l] += self.weights[j, k, l, m] * d_l_d_z * d_l_d_A[i, m]
        self.weights -= learning_rate * d_l_d_W
        self.biases -= learning_rate * d_l_d_b
        return d_l_d_input


class DenseLayer(DenseLayerBase):

    def __init__(self,
                 input_shape=[-1, 7, 7, 64],
                 output_shape=[-1, 10],
                 activation=activation_none):
        super().__init__(input_shape, output_shape, activation)
        self.weights = self.weights.reshape(-1, self.output_shape[1])

    def forward(self, input_):
        self.input_reshaped = input_.reshape([self.input_shape[0], -1])
        # A = XW + (bs(1^T))^T
        output = self.activation.activate(self.input_reshaped.dot(self.weights) + self.biases)
        self.last_output = output
        return output

    def backprop(self, d_l_d_A, learning_rate=0.001):
        d_l_d_Z = np.multiply(d_l_d_A, self.activation.derive(self.last_output))
        # dl = tr[(dl/dA)^T dA] = tr[(dl/dA)^T XdW] = tr{[X^T (dl/dA)]^T dW}
        d_l_d_W = self.input_reshaped.T.dot(d_l_d_Z)
        # dl = tr[(dl/dA)^T dA] = tr[(dl/dA)^T (dbs 1^T)^T]
        #    = tr[dbs 1^T (dl/dA)]
        #    = tr[1^T (dl/dA) dbs]
        #    = tr{[(dl/dA)^T 1]^T dbs}
        d_l_d_b = np.sum(d_l_d_Z.T, axis=1)
        # dl = tr[(dl/dA)^T dA] = tr[(dl/dA)^T dX W] = tr{[W (dl/dA)]^T dX}
        #    = tr{[(dl/dA) W^T]^T dX}
        d_l_d_input = d_l_d_Z.dot(self.weights.T).reshape(self.input_shape)
        self.weights -= learning_rate * d_l_d_W
        self.biases -= learning_rate * d_l_d_b
        return d_l_d_input

#test_layer(DenseLayerBase, DenseLayer)
