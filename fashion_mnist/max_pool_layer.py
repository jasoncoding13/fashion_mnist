# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:16:52 2019

@author: jason
"""

import numpy as np


class MaxPoolLayerBase():

    def __init__(self,
                 input_shape=[-1, 28, 28, 32],
                 pool_shape=[2, 2],
                 strides=[2, 2]):
        # [batch_size, in_height, in_width, in_channels]
        self.input_shape = np.array(input_shape)
        # [pool_height, pool_width]
        self.pool_shape = np.array(pool_shape)
        # [height_stride, width_stride]
        self.strides = np.array(strides)

    def compute_shape(self, batch_size):
        self.input_shape[0] = batch_size
        # output shape with `valid padding`
        # [batch_size, out_height, out_width, out_channels]
        self.output_shape = np.r_[self.input_shape[0],
                                  ((self.input_shape[1:3] - self.pool_shape) // self.strides + 1).astype(int),
                                  self.input_shape[3]]

    def _gen_patches(self, input_):
        # generate patch based on given sample and in_channel
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        # [pool_height, pool_width]
                        yield i, j, k, l, input_[i,
                                                 j*self.strides[0]:j*self.strides[0]+self.pool_shape[0],
                                                 k*self.strides[1]:k*self.strides[1]+self.pool_shape[1],
                                                 l]

    def forward(self, input_):
        self.input_ = input_
        output = np.zeros(self.output_shape)
        for i, j, k, l, patch in self._gen_patches(input_):
            output[i, j, k, l] = np.amax(patch)
        self.output = output
        return output

    def backprop(self, d_l_d_A, learning_rate):
        d_l_d_input = np.zeros(self.input_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.output_shape[2]):
                    for l in range(self.output_shape[3]):
                        index_nonzero = np.nonzero(
                                self.input_[i,
                                            j*self.strides[0]:j*self.strides[0]+self.pool_shape[0],
                                            k*self.strides[1]:k*self.strides[1]+self.pool_shape[1],
                                            l] == self.output[i, j, k, l])
                        index_nonzero = (np.array(index_nonzero[0][0]),
                                         np.array(index_nonzero[1][0]))
#                        index_argmax = np.argmax(
#                                self.input_[i,
#                                            j*self.strides[0]:j*self.strides[0]+self.pool_shape[0],
#                                            k*self.strides[1]:k*self.strides[1]+self.pool_shape[1],
#                                            l])
#                        index_argmax = np.unravel_index(index_argmax, self.pool_shape)
#                        assert np.equal(index_nonzero, index_argmax).all()
                        d_l_d_input[i,
                                    j*self.strides[0]:j*self.strides[0]+self.pool_shape[0],
                                    k*self.strides[1]:k*self.strides[1]+self.pool_shape[1],
                                    l][index_nonzero] += d_l_d_A[i, j, k, l]
        return d_l_d_input


class MaxPoolLayer(MaxPoolLayerBase):

    def _gen_patches(self, input_):
        # genereate patches besed on out_height, out_width
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                # [batch_size, pool_height, pool_width, in_channels]
                yield i, j, input_[:,
                                   i*self.strides[0]:i*self.strides[0]+self.pool_shape[0],
                                   j*self.strides[1]:j*self.strides[1]+self.pool_shape[1],
                                   :]

    def forward(self, input_):
#        self.input_ = input_
        output = np.zeros(self.output_shape)
        self.index_seq = [[],[],[],[]]
        for i, j, patch in self._gen_patches(input_):
            output[:, i, j, :] = np.amax(patch, axis=(1, 2))
            index_argmax = np.argmax(
                    patch.reshape(
                            self.input_shape[0], np.prod(self.pool_shape), self.input_shape[3]), axis=1)
            index_argmax = np.array(np.unravel_index(index_argmax, shape=self.pool_shape))
            for k in range(self.input_shape[0]):
                for l in range(self.input_shape[3]):
                    self.index_seq[0].append(k)
                    self.index_seq[1].append(i*self.strides[0]+index_argmax[0, k, l])
                    self.index_seq[2].append(j*self.strides[1]+index_argmax[1, k, l])
                    self.index_seq[3].append(l)
#        self.output = output
        return output

    def backprop(self, d_l_d_A, learning_rate):
        d_l_d_input = np.zeros(self.input_shape)
        # [1, 2, 0, 3] is besed on the order of loop in forward
        d_l_d_input[self.index_seq[0],
                    self.index_seq[1],
                    self.index_seq[2],
                    self.index_seq[3]] = np.transpose(d_l_d_A, [1, 2, 0, 3]).reshape(-1)
        return d_l_d_input

#test_layer(MaxPoolLayerBase, MaxPoolLayer)
