#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:16:53 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np


class activation_relu():

    def activate(self, input_):
        return np.maximum(input_, 0)

    def derive(self, output):
        return output > 0


class activation_none():
    def activate(self, input_):
        return input_

    def derive(self, output):
        return 1
