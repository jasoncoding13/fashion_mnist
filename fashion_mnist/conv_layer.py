# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:20:04 2019

@author: jason
"""

import math
import numpy as np
from .activation import activation_none, activation_relu
from .utils import get_variables, test_layer


class ConvLayerBase():

    def __init__(
            self,
            input_shape=[-1, 28, 28, 1],
            filter_shape=[5, 5, 1, 32],
            strides=[1, 1],
            activation=activation_none):
        # [batch_size, in_height, in_width, in_channels]
        self.input_shape = np.array(input_shape)
        # [filter_height, filter_width, in_channels, out_channels]
        self.filter_shape = np.array(filter_shape)
        # [height_stride, width_stride]
        self.strides = np.array(strides)
        limit = math.sqrt(6 / np.prod(self.filter_shape[0:3]))
        self.filters = get_variables(self.filter_shape, loc=-limit, scale=2*limit)
        self.biases = np.zeros(self.filter_shape[3])
        self.activation = activation()

    def compute_shape(self, batch_size):
        self.input_shape[0] = batch_size
        # output shape with `same padding`
        # [batch_size, out_height, out_width, out_channels]
        self.output_shape = np.r_[self.input_shape[0],
                                  np.ceil(self.input_shape[1:3] / self.strides).astype(int),
                                  self.filter_shape[3]]

        # (output_shape[d] - 1 ) * strides[d] + filter_shape[d]
        # >= input_shape[d] = input_shape[d] + padding_sizes[d]
        # d : the dimension of height or width
        self.padding_sizes = ((self.output_shape[1:3] - 1) * self.strides +
                              self.filter_shape[0:2] - self.input_shape[1:3])

        # [batch_size, padded_height, padded_width, in_channels]
        self.padded_shape = np.r_[self.input_shape[0],
                                  self.input_shape[1:3] + self.padding_sizes,
                                  self.input_shape[3]]

        # indexes from left or up edge to place raw input
        # odd padding_sizes | even padding_sizes
        # 0 0 0 0           | 0 0 0
        # 0 1 0 0           | 0 1 0
        # 0 0 0 0           | 0 0 0
        # 0 0 0 0
        self.start_indexes = self.padding_sizes//2
        self.end_indexes = self.start_indexes + self.input_shape[1:3]

    def _pad_input(self, input_):
        input_padded = np.zeros(self.padded_shape)
        input_padded[:,
                     self.start_indexes[0]:self.end_indexes[0],
                     self.start_indexes[1]:self.end_indexes[1],
                     :] = input_
        return input_padded

    def _gen_patches(self, input_):
        # generate patch based on given sample
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    # [filter_height, filter_width, in_channels]
                    yield i, j, k, input_[i,
                                          j*self.strides[0]:j*self.strides[0]+self.filter_shape[0],
                                          k*self.strides[1]:k*self.strides[1]+self.filter_shape[1],
                                          :]

    def forward(self, input_):
        self.padded_input = self._pad_input(input_)
        output = np.zeros(self.output_shape)
        for i, j, k, patch in self._gen_patches(self.padded_input):
            # output channels
            for l in range(self.output_shape[3]):
                # P: patch from padded_input [filter_height, filter_width, in_channels]
                # F: filter for l-th out_channel [filter_height, filter_width, in_channels]
                # b: bias for l-th out_channel
                # here are totally `out_channels` filters and `out_channels` biases
                # z = tr[P^T F] + b
                # a = relu(z): output for i-th sample j-th out_height k-th out_width l-th out_channel
                output[i, j, k, l] = self.activation.activate(np.sum(np.multiply(patch, self.filters[:, :, :, l]))+self.biases[l])
        self.last_output = output
        return output

    def backprop(self, d_l_d_A, learning_rate=0.001):
        # shape of d_l_d_A is equal to output_shape
        d_l_d_filters = np.zeros(self.filter_shape)
        d_l_d_biases = np.zeros(self.filter_shape[3])
        d_l_d_input_padded = np.zeros_like(self.padded_input)
        d_l_d_input = np.zeros(self.input_shape)
        # da   1 z>0
        # -- =
        # dz   0 z<=0 here we prefer sparser

        # dz = dtr(P^T F) = tr(P^T dF)
        # dz = dtr(P^T F) = tr[(dP)^T F] = tr(F^T dP)
        d_l_d_Z = np.multiply(d_l_d_A, self.activation.derive(self.last_output))
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        d_l_d_filters[:, :, :, l] += (d_l_d_Z[i, j, k, l] *
                                     self.padded_input[i,
                                                       j*self.strides[0]:j*self.strides[0]+self.filter_shape[0],
                                                       k*self.strides[1]:k*self.strides[1]+self.filter_shape[1],
                                                       :])
                        d_l_d_biases[l] += d_l_d_Z[i, j, k, l]
                        d_l_d_input_padded[i,
                                    j*self.strides[0]:j*self.strides[0]+self.filter_shape[0],
                                    k*self.strides[1]:k*self.strides[1]+self.filter_shape[1],
                                    :] += (d_l_d_Z[i, j, k, l] * self.filters[:, :, :, l])

        # Cut d_l_d_input_padded fot d_l_input
        # 0 0 0
        # 0 1 0 > 1
        # 0 0 0
        for m in range(self.padded_shape[1]):
            for n in range(self.padded_shape[2]):
                # index in raw input
                m_ = m - self.start_indexes[0]
                n_ = n - self.start_indexes[1]
                if (m_ >= 0) and (m_ < self.input_shape[1]) and \
                        (n_ >= 0) and (n_ < self.input_shape[2]):
                    d_l_d_input[:,
                                m_,
                                n_,
                                :] = d_l_d_input_padded[:, m, n, :]

        self.filters -= learning_rate * d_l_d_filters
        self.biases -= learning_rate * d_l_d_biases
        return d_l_d_input


class ConvLayer(ConvLayerBase):

    def _gen_patches(self, input_):
        # genereate patches besed on out_height, out_width
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                # [batch_size, filter_height, filter_width, in_channels]
                yield i, j, input_[:,
                                   i*self.strides[0]:i*self.strides[0]+self.filter_shape[0],
                                   j*self.strides[1]:j*self.strides[1]+self.filter_shape[1],
                                   :]

    def forward(self, input_):
        self.padded_input = self._pad_input(input_)
        output = np.zeros(self.output_shape)
        for i, j, patch in self._gen_patches(self.padded_input):
            # X_r: patches reshaped [batch_size, filter_height * filter_width * in_channels]
            # W: filters reshaped [filter_height * filter_width * in_channels, out_channels]
            # bs: biases

            # (bs(1^T))^T:
            # [out_channels, 1] * [1, batch_size] > [out_channels, batch_size]
            # b_1, b_1, ..., b_1 |transpose it > [batch_size, out_channels]
            # b_2, b_2, ..., b_2 |o: abbreviation for `out_channels`
            # ..., ..., ..., ... |
            # b_o, b_o, ..., b_o |

            # Z = X_r W + (bs(1^T))^T
            # A = relu(Z): outputs for i-th out_height, j-th out_width [batch_size, out_channels]
            output[:, i, j, :] = self.activation.activate(
                    patch.reshape(
                            self.input_shape[0],
                            np.prod(self.filter_shape[0:3])).dot(
                                    self.filters.reshape(
                                            np.prod(
                                                    self.filter_shape[0:3]),
                                                    self.filter_shape[3])) + self.biases)
        self.last_output = output
        return output

    def backprop(self, d_l_d_A, learning_rate=0.001):
        d_l_d_filters = np.zeros(self.filter_shape)
        d_l_d_biases = np.zeros(self.filter_shape[3])
        d_l_d_input_padded = np.zeros_like(self.padded_input)
        d_l_d_input = np.zeros(self.input_shape)
        # shape of d_l_d_A is equal to output_shape
        # For each patch,

        # dl = tr[(dl/dA)^T dA] = tr{(dl/dA)^T [relu'(X_r W+b) element-* dZ]}
        #    = tr{(dl/dA)^T [relu'(Z) element-* (X_r dW)]}
        #    = tr{[(dl/dA) element-* relu'(Z)]^T X_r dW}
        #    = tr{[(X_r)^T (dl/dA) element-* relu'(Z)]^T dW

        # dl = tr[(dl/dA)^T dA] = tr{(dl/dA)^T [relu'(X_r W+b) element-* dZ]}
        #    = tr{(dl/dA)^T [relu'(X_r W+b) element-* (dbs 1^T)^T]}
        #    = tr{[(dl/dA) element-* relu'(Z)]^T (dbs 1^T)^T}
        #    = tr[dbs 1^T (dl/dA) element-* relu'(Z)]
        #    = tr[1^T (dl/dA) element-* relu'(Z) dbs]
        #    = tr< {[(dl/dA) element-* relu'(Z)]^T 1}^T dbs >
        #    here <> means brackets bigger than {}
        # dl = tr[(dl/dA)^T dA] = tr{(dl/dA)^T [relu'(X_r W+b) element-* dZ]}
        #    = tr{(dl/dA)^T [relu'(Z) element-* (dX_r W)]}
        #    = tr{[(dl/dA) element-* relu'(Z)]^T dX_r W}
        #    = tr{W [(dl/dA) element-* relu'(Z)]^T dX_r}
        #    = tr{(dl/dA) element-* relu'(Z) (W)^T]^T dX_r}
        for i, j, patch in self._gen_patches(self.padded_input):
            # d_l_d_Z: [batch_size, out_channels]
            d_l_d_Z = np.multiply(d_l_d_A[:, i, j, :], self.activation.derive(self.last_output[:, i, j, :]))
            d_l_d_filters += patch.reshape([self.input_shape[0], -1]).T.dot(d_l_d_Z).reshape(self.filter_shape)
            d_l_d_biases += np.sum(d_l_d_Z.T, axis=1)
            d_l_d_input_padded[:,
                        i*self.strides[0]:i*self.strides[0]+self.filter_shape[0],
                        j*self.strides[1]:j*self.strides[1]+self.filter_shape[1],
                        :] += d_l_d_Z.dot(
                        self.filters.reshape(
                                np.prod(self.filter_shape[0:3]),
                                        self.filter_shape[3]).T).reshape(np.r_[self.input_shape[0],
                                                                               self.filter_shape[0:3]])
        d_l_d_input = d_l_d_input_padded[:,
                                         self.start_indexes[0]:self.end_indexes[0],
                                         self.start_indexes[1]:self.end_indexes[1],
                                         :]
        self.filters -= learning_rate * d_l_d_filters
        self.biases -= learning_rate * d_l_d_biases
        return d_l_d_input

#test_layer(ConvLayerBase, ConvLayer, 'conv')
