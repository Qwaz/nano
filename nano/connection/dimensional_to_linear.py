from abc import ABCMeta
import math

import numpy as np

from nano.connection import AbsConnection
from nano.layer import DimensionalLayer
from nano.layer import LinearLayer


class DimensionToLinear(AbsConnection, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def check_layer_type(self, before_layer, after_layer):
        return isinstance(before_layer, DimensionalLayer) and isinstance(after_layer, LinearLayer)


class Projection(DimensionToLinear):
    def __init__(self):
        super().__init__()

    def check_shape(self, before_shape, after_shape):
        return True

    def prepare_connection(self, before_shape, after_shape):
        self.weight.append(
            np.random.randn(before_shape[0] * before_shape[1] * before_shape[2], after_shape[1]) *
            math.sqrt(2 / (before_shape[0] * before_shape[1] * before_shape[2]))
        )
        self.weight.append(
            np.zeros(after_shape[1])
        )

    def forward(self):
        self.after_layer.result += np.dot(self.before_layer.result.flatten(), self.weight[0]) + self.weight[1]

    def backward(self):
        before_size = self.before_layer.shape[0] * self.before_layer.shape[1] * self.before_layer.shape[2]
        dw = np.repeat(np.reshape(self.before_layer.result, (before_size, 1)), self.after_layer.shape[1], axis=1)
        self.dweight[0][:] = np.multiply(dw, self.after_layer.error)
        self.dweight[1][:] = self.after_layer.error

        de = np.multiply(self.weight[0], self.after_layer.error)
        de = np.sum(de, axis=1)
        self.before_layer.error[:] += np.reshape(de, self.before_layer.shape)
