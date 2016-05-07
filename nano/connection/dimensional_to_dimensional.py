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
    def __init__(self, height, width, padding=0):
        super().__init__()
        self.height = height
        self.width = width
        self.padding = padding
        self.stride = 1 # currently nano only supports stride=1

    def check_shape(self, before_shape, after_shape):
        pad_height = before_shape[1] + 2 * self.padding
        pad_width = before_shape[2] + 2 * self.padding

        # check minimum size
        if pad_width < self.width or pad_height < self.height:
            return False

        # check exact fit
        if (pad_height - self.height) % self.stride != 0 or (pad_width - self.width) % self.stride != 0:
            return False

        # check size
        return (
            (pad_height - self.height) // self.stride + 1 == after_shape[1]
            and (pad_width - self.width) // self.stride + 1 == after_shape[2]
        )

    def prepare_connection(self, before_shape, after_shape):
        self.depth = before_shape[0]
        self.filters = after_shape[0]

        # filters (intialization for ReLU)
        self.weight.append(
            np.random.randn(self.filters, self.depth, self.height, self.width)
            * math.sqrt(2 / (before_shape[0] * before_shape[1] * before_shape[2]))
        )
        # biases
        self.weight.append(
            np.zeros(self.filters)
        )

    def forward(self):
        # im2col implementation
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]

        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
        padded = np.pad(self.before_layer.result, npad, mode='constant', constant_values=0)

        row = np.zeros((self.filters, self.depth * self.height * self.width))
        col = np.zeros((self.depth * self.height * self.width, result_height * result_width))

        for f in range(self.filters):
            row[f, :] = self.weight[0][f].flatten()

        for loc_y in range(result_height):
            for loc_x in range(result_width):
                index = loc_x + result_width * loc_y
                col[:, index] = padded[
                    :,
                    loc_y * self.stride : loc_y * self.stride + self.height,
                    loc_x * self.stride : loc_x * self.stride + self.width
                ].flatten()

        self.after_layer.result += np.reshape(np.dot(row, col), self.after_layer.shape)

        # bias addition
        self.after_layer.result += self.weight[1].reshape(self.filters, 1, 1)

    def backward(self):
        '''
        only strides = 1 is supported
        '''
        # error propagation
        before_height = self.before_layer.shape[1]
        before_width = self.before_layer.shape[2]

        weight_t = np.transpose(self.weight[0], (1, 0, 3, 2))

        npad = ((0, 0), (self.height-1, self.height-1), (self.width-1, self.width-1))
        padded = np.pad(self.after_layer.error, npad, mode='constant', constant_values=0)

        row = np.zeros((self.depth, self.filters * self.height * self.width))
        col = np.zeros((self.filters * self.height * self.width, before_height * before_width))

        for f in range(self.depth):
            row[f, :] = weight_t[f].flatten()

        for loc_y in range(before_height):
            for loc_x in range(before_width):
                index = loc_x + before_width * loc_y
                col[:, index] = padded[:, loc_y : loc_y + self.height, loc_x : loc_x + self.width].flatten()

        self.before_layer.error += np.reshape(np.dot(row, col), self.before_layer.shape)

        # bias propagation
        self.dweight[1][:] = np.sum(self.after_layer.error, axis=(1,2)).flat

        # weight propagation
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]

        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
        padded = np.pad(self.before_layer.result, npad, mode='constant', constant_values=0)

        for f in range(self.filters):
            row = np.zeros((self.depth, self.depth * result_height * result_width))
            col = np.zeros((self.depth * result_height * result_width, self.height * self.width))

            for d in range(self.depth):
                row[d, :] = np.repeat(self.after_layer.error[f].flatten(), self.depth)
                
            for loc_y in range(self.height):
                for loc_x in range(self.width):
                    index = loc_x + self.width * loc_y
                    col[:, index] = padded[:, loc_y : loc_y + result_height, loc_x : loc_x + result_width].flatten()

            self.dweight[0][f][:] = np.reshape(np.dot(row, col), (self.depth, self.height, self.width))

class MaxPooling(DimensionalToDimensional):
    def __init__(self, size, stride):
        super().__init__()
        self.size = size
        self.stride = stride

    def check_shape(self, before_shape, after_shape):
        # check exact fit
        if (before_shape[1] - self.size) % self.stride != 0 or (before_shape[2] - self.size) % self.stride != 0:
            return False

        # check size
        return (
            before_shape[0] == after_shape[0]
            and (before_shape[1] - self.size) // self.stride + 1 == after_shape[1]
            and (before_shape[2] - self.size) // self.stride + 1 == after_shape[2]
        )

    def prepare_connection(self, before_shape, after_shape):
        pass

    def forward(self):
        result_depth = self.after_layer.shape[0]
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]

        for f in range(result_depth):
            for loc_y in range(result_height):
                for loc_x in range(result_width):
                    self.after_layer.result[f, loc_y, loc_x] += (
                        np.amax(self.before_layer.result[
                            f,
                            loc_y * self.stride : loc_y * self.stride + self.size,
                            loc_x * self.stride : loc_x * self.stride + self.size
                        ])
                    )

    def backward(self):
        result_depth = self.after_layer.shape[0]
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]

        for f in range(result_depth):
            for loc_y in range(result_height):
                for loc_x in range(result_width):
                    index = np.argmax(self.before_layer.result[
                        f,
                        loc_y * self.stride : loc_y * self.stride + self.size,
                        loc_x * self.stride : loc_x * self.stride + self.size
                    ])
                    self.before_layer.error[f].flat[index] += self.after_layer.error[f, loc_y, loc_x]

class AveragePooling(DimensionalToDimensional):
    def __init__(self, size, stride):
        super().__init__()
        self.size = size
        self.stride = stride

    def check_shape(self, before_shape, after_shape):
        # check exact fit
        if (before_shape[1] - self.size) % self.stride != 0 or (before_shape[2] - self.size) % self.stride != 0:
            return False

        # check size
        return (
            before_shape[0] == after_shape[0]
            and (before_shape[1] - self.size) // self.stride + 1 == after_shape[1]
            and (before_shape[2] - self.size) // self.stride + 1 == after_shape[2]
        )

    def prepare_connection(self, before_shape, after_shape):
        pass

    def forward(self):
        result_depth = self.after_layer.shape[0]
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]

        row = np.zeros((self.size ** 2))
        col = np.zeros((self.size ** 2, result_depth * result_height * result_width))

        row.fill(1 / (self.size ** 2))

        for d in range(result_depth):
            for loc_y in range(result_height):
                for loc_x in range(result_width):
                    index = loc_x + result_width * loc_y + result_height * result_width * d
                    col[:, index] = self.before_layer.result[
                        d,
                        loc_y * self.stride : loc_y * self.stride + self.size,
                        loc_x * self.stride : loc_x * self.stride + self.size
                    ].flatten()

        self.after_layer.result += np.reshape(np.dot(row, col), self.after_layer.shape)

    def backward(self):
        result_depth = self.after_layer.shape[0]
        result_height = self.after_layer.shape[1]
        result_width = self.after_layer.shape[2]
        error_matrix = np.empty((self.size, self.size))

        for f in range(result_depth):
            for loc_y in range(result_height):
                for loc_x in range(result_width):
                    error_matrix.fill(self.after_layer.error[f, loc_y, loc_x] / (self.size ** 2))
                    (self.before_layer.error[
                        f,
                        loc_y * self.stride : loc_y * self.stride + self.size,
                        loc_x * self.stride : loc_x * self.stride + self.size
                    ]) = error_matrix
