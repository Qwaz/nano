from abc import ABCMeta

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
        self.weight.append(
            2 * np.random.random((before_shape[1], after_shape[1])) - 1# weight
        )
        self.weight.append(
            np.random.random(after_shape[1]) # bias
        )

    def forward(self):
        self.after_layer.result += np.dot(self.before_layer.result, self.weight[0]) + self.weight[1]

    def backward(self):
        dw = np.repeat(self.before_layer.result.T, self.after_layer.shape[1], axis=1)
        dw = np.multiply(dw, self.after_layer.error)
        np.copyto(self.dweight[0], dw)
        np.copyto(self.dweight[1], self.after_layer.error)

        de = np.multiply(self.weight[0], self.after_layer.error)
        de = np.sum(de, axis=1)
        np.copyto(self.before_layer.error, de.T)
