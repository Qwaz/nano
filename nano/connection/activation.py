from abc import ABCMeta
import math

import numpy as np

from nano.connection import AbsConnection


class AbsActivation(AbsConnection, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.fx = np.vectorize(lambda x: x)
        self.dffx = np.vectorize(lambda x: x)

    def check_layer_type(self, before_layer, after_layer):
        return True

    def check_shape(self, before_shape, after_shape):
        return before_shape == after_shape

    def forward(self):
        self.after_layer.result += self.fx(self.before_layer.result)

    def backward(self):
        self.before_layer.error += np.multiply(self.after_layer.error, self.dfdx(self.before_layer.result))


class Linear(AbsActivation):
    def __init__(self):
        super().__init__()

        self.fx = np.vectorize(lambda x: x)
        self.dfdx = np.vectorize(lambda x: 1)


class ReLU(AbsActivation):
    def __init__(self):
        super().__init__()

        self.fx = np.vectorize(lambda x: x * (x > 0))
        self.dfdx = np.vectorize(lambda x: x > 0)


class LeakyReLU(AbsActivation):
    def __init__(self, leak=0.01):
        super().__init__()

        self.fx = np.vectorize(lambda x: x if x > 0 else leak * x)
        self.dfdx = np.vectorize(lambda x: 1 if x > 0 else leak)


class Sigmoid(AbsActivation):
    def __init__(self):
        super().__init__()

        self.fx = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
        self.dfdx = np.vectorize(lambda x: (1 / (1 + math.exp(-x))) * (1 - 1 / (1 + math.exp(-x))))


class Tansig(AbsActivation):
    def __init__(self):
        super().__init__()

        self.fx = np.vectorize(lambda x: math.tanh(x))
        self.dfdx = np.vectorize(lambda x: 1 - (math.tanh(x) ** 2))


class Softmax(AbsActivation):
    def __init__(self):
        super().__init__()

    def forward(self):
        norm_result = np.copy(self.before_layer.result)
        norm_result -= np.amax(norm_result)
        exp_scores = np.exp(norm_result)
        self.after_layer.result += exp_scores / np.sum(exp_scores)

    def backward(self):
        self.before_layer.error += self.after_layer.error
