from abc import ABCMeta
import math

import numpy as np

from nano.connection import AbsConnection, abstractmethod


class AbsActivation(AbsConnection, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def check_layer_type(self, before_layer, after_layer):
        return True

    def check_shape(self, before_shape, after_shape):
        return before_shape == after_shape

    def forward(self):
        self.after_layer.result += type(self).fx(self.before_layer.result)

    def backward(self):
        self.before_layer.error += np.multiply(self.after_layer.error, type(self).dfdx(self.before_layer.result))

class Linear(AbsActivation):
    def __init__(self):
        super().__init__()

    fx = np.vectorize(lambda x: x)
    dfdx = np.vectorize(lambda x: 1)

class ReLU(AbsActivation):
    def __init__(self):
        super().__init__()

    fx = np.vectorize(lambda x: x * (x > 0))
    dfdx = np.vectorize(lambda x: x > 0)

class Sigmoid(AbsActivation):
    def __init__(self):
        super().__init__()

    fx = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    dfdx = np.vectorize(lambda x: (1 / (1 + math.exp(-x))) * (1 - 1 / (1 + math.exp(-x))))

class Tansig(AbsActivation):
    def __init__(self):
        super().__init__()

    fx = np.vectorize(lambda x: math.tanh(x))
    dfdx = np.vectorize(lambda x: 1 - (math.tanh(x) ** 2))

class Softmax(AbsActivation):
    def __init__(self):
        super().__init__()

    def forward(self):
        exp_scores = np.exp(self.before_layer.result)
        self.after_layer.result += exp_scores / np.sum(exp_scores)

    def backward(self):
        self.before_layer.error += self.after_layer.error
