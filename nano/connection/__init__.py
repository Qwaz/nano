from abc import ABCMeta, abstractmethod

import numpy as np


class ConnectionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AbsConnection(metaclass=ABCMeta):
    def __init__(self):
        self.connected = False
        self.weight = []
        self.dweight = []

    def connect(self, before_layer, after_layer):
        if self.connected:
            raise ConnectionError("The connection is already used")

        if not self.check_layer_type(before_layer, after_layer):
            raise ConnectionError("Layer type mismatch")

        if self.check_shape(before_layer.shape, after_layer.shape):
            before_layer.connections.append(self)
            after_layer.pre_layer_count += 1

            self.before_layer = before_layer
            self.after_layer = after_layer

            self.connected = True
            self.prepare_connection(before_layer.shape, after_layer.shape)

            # initialize dweight
            for arr in self.weight:
                self.dweight.append(np.empty(arr.shape))
        else:
            raise ConnectionError("Layer shape mismatch")

    @abstractmethod
    def check_layer_type(self, before_layer, after_layer):
        pass

    @abstractmethod
    def check_shape(self, before_shape, after_shape):
        pass

    def prepare_connection(self, before_shape, after_shape):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
