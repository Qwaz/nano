from abc import ABCMeta
import math

import numpy as np

from nano.connection import AbsConnection
from nano.layer import LinearLayer


class LinearToLinear(AbsConnection, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def check_layer_type(self, before_layer, after_layer):
        return isinstance(before_layer, LinearLayer) and isinstance(after_layer, LinearLayer)

class FullyConnected(LinearToLinear):
    def __init__(self):
        super().__init__()

    def check_shape(self, before_shape, after_shape):
        return True

    def prepare_connection(self, before_shape, after_shape):
        # weight (intialization for ReLU)
        self.weight.append(
            (2 * np.random.randn(before_shape[1], after_shape[1]) - 1)
            * math.sqrt(2 / before_shape[1])
        )
        # bias
        self.weight.append(
            np.zeros(after_shape[1])
        )

    def forward(self):
        self.after_layer.result += np.dot(self.before_layer.result, self.weight[0]) + self.weight[1]

    def backward(self):
        dw = np.repeat(self.before_layer.result.T, self.after_layer.shape[1], axis=1)
        self.dweight[0][:] = np.multiply(dw, self.after_layer.error)
        self.dweight[1][:] = self.after_layer.error

        de = np.multiply(self.weight[0], self.after_layer.error)
        self.before_layer.error[:] = np.sum(de, axis=1).T
