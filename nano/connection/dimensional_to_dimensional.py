from abc import ABCMeta
import math

import numpy as np

from nano.connection import AbsConnection
from nano.layer import DimensionalLayer


class DimensionalToDimensional(AbsConnection, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def check_layer_type(self, before_layer, after_layer):
        return isinstance(before_layer, DimensionalLayer) and isinstance(after_layer, DimensionalLayer)

class Convolution(DimensionalToDimensional):
    def __init__(self, height, width, stride=1):
        super().__init__()
        self.stride = stride
        self.height = height
        self.width = width

    def check_shape(self, before_shape, after_shape):
        if ((before_shape[1] - self.height) // self.stride + 1 == after_shape[1]
            and (before_shape[2] - self.width) // self.stride + 1 == after_shape[2]):
            return True
        else:
            return False

    def prepare_connection(self, before_shape, after_shape):
        self.depth = before_shape[0]
        self.filters = after_shape[0]
        for i in range(self.filters):
            # filters
            self.weight.append(
                (2 * np.random.randn(self.depth, self.height, self.width) - 1)
                * math.sqrt(2 / (before_shape[0] * before_shape[1] * before_shape[2]))
                # intialization for ReLU
            )
        # biases
        self.weight.append(
            np.zeros(self.filters)
        )

    def forward(self):
        for i in range(self.filters):
            filter_vec = self.weight[i].flatten()
            bias = np.array([self.weight[self.filters][i]])

            calc_matrix = np.zeros((self.height * self.width * self.depth , self.after_layer.shape[1] * self.after_layer.shape[2]))
            for j in range(self.after_layer.shape[1]):
                for k in range(self.after_layer.shape[2]):
                    temp = np.zeros((self.height * self.width * self.depth, 1))
                    for d in range(self.depth):
                        layer_values = self.before_layer.result[d, j * self.stride : j * self.stride + self.height, k * self.stride : k * self.stride + self.width]
                        layer_values = layer_values.reshape((self.height * self.width , 1))
                        temp[d * self.height * self.width : (d + 1) * self.height * self.width, :] = layer_values
                    calc_matrix[:, j * self.after_layer.shape[2] + k : j * self.after_layer.shape[2] + k + 1] = temp

            res = np.dot(filter_vec, calc_matrix)
            res[:] = res + np.tile(bias, self.after_layer.shape[1] * self.after_layer.shape[2])
            res = res.reshape((self.after_layer.shape[1], self.after_layer.shape[2]))
            self.after_layer.result[i, :, :] += res

    def backward(self):
        '''
        backprop currently supports only strides of size 1!!!
        '''
        back_error_prop = np.zeros((self.depth, self.before_layer.shape[1], self.before_layer.shape[2]))

        # zero padding for size of matrix
        for i in range(self.filters):
            temp_weights = np.zeros((self.after_layer.shape[1] + (self.height - 1) * 2, self.after_layer.shape[2] * (self.width - 1) * 2))
            temp_weights[self.height - 1 : self.height + self.after_layer.shape[1] - 1, self.width - 1 : self.width + self.after_layer.shape[2] - 1] = self.after_layer.error[i, :, :]
            # create the matrix of errors
            error_matrix = np.zeros((self.height * self.width, self.before_layer.shape[1] * self.before_layer.shape[2]))

            for j in range(self.before_layer.shape[1]):
                for k in range(self.before_layer.shape[2]):
                    temp = temp_weights[j * self.stride : j * self.stride + self.height , k * self.stride : k * self.stride + self.width]
                    temp = temp.reshape(self.height * self.width, 1)
                    error_matrix[:, j * self.before_layer.shape[2] + k : j * self.before_layer.shape[2] + k + 1] = temp

            for j in range(self.depth):
                filter_vec = self.weight[i][j, :, :].flatten()
                filter_vec = filter_vec[: : -1]
                back_error = np.dot(filter_vec, error_matrix)
                back_error = back_error.reshape(self.before_layer.shape[1], self.before_layer.shape[2])
                back_error_prop[j, :, :] += back_error

        self.before_layer.error[:] += back_error_prop # this transmits the error

        for i in range(self.filters):
            filter_vec = self.after_layer.error[i, :, :] # this is the filter for the convolution (backprop uses the error as a convolution filter)
            filter_vec = filter_vec.flatten() # flatten it
            bias_update = filter_vec.sum()

            calc_matrix = np.zeros((self.after_layer.shape[1] * self.after_layer.shape[2], self.depth * self.height * self.width))
            temp = np.zeros((self.after_layer.shape[1] * self.after_layer.shape[2], 1))
            for d in range(self.depth):
                for j in range(self.height):
                    for k in range(self.width):
                        layer_values = self.before_layer.result[d, j * self.stride : j * self.stride + self.after_layer.shape[1], k * self.stride : k * self.stride + self.after_layer.shape[2]]
                        layer_values = layer_values.reshape(self.after_layer.shape[1] * self.after_layer.shape[2], 1)
                        temp = layer_values
                        calc_matrix[:, d * self.height * self.width + j * self.width + k : d * self.height * self.width + j * self.width + k + 1] = temp

            error_matrix = np.dot(filter_vec, calc_matrix)
            error_matrix = error_matrix.reshape(self.depth, self.height, self.width)
            self.dweight[i] = error_matrix # update filter
            self.dweight[self.filters][i] = bias_update

class MaxPooling(DimensionalToDimensional):
    def __init__(self, height, width, stride):
        super().__init__()
        self.height = height
        self.width = width
        self.stride = stride

    def check_shape(self, before_shape, after_shape):
        if (before_shape[0] == after_shape[0]
            and (before_shape[1] - self.height) // self.stride + 1 == after_shape[1]
            and (before_shape[2] - self.width) // self.stride + 1 == after_shape[2]):
            return True
        else:
            return False

    def prepare_connection(self, before_shape, after_shape):
        pass

    def forward(self):
        for i in range(self.after_layer.shape[0]):
            for j in range(self.after_layer.shape[1]):
                for k in range(self.after_layer.shape[2]):
                    self.after_layer.result[i, j, k] += (
                        np.amax(self.before_layer.result[i, self.stride * j : self.stride * j + self.height, self.stride * k : self.stride * k + self.width]))

    def backward(self):
        self.before_layer.error.fill(0)
        for i in range(self.after_layer.shape[0]):
            for j in range(self.after_layer.shape[1]):
                for k in range(self.after_layer.shape[2]):
                    ind = np.unravel_index(np.argmax(self.before_layer.result[i, self.stride * j : self.stride * j + self.height, self.stride * k : self.stride * k + self.width]), (1, self.height, self.width))
                    d = ind[0] + i
                    h = ind[1] + self.stride * j
                    w = ind[2] + self.stride * k
                    self.before_layer.error[d, h, w] += self.after_layer.error[i, j, k]
