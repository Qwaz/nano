from abc import ABCMeta

import numpy as np


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self.added = False
        self.connections = []
        self.pre_layer_count = 0 # overall connection count
        self.pre_layer_connect = 0 # connection count for topology sort

class LinearLayer(Layer):
    def __init__(self, num_neurons):
        Layer.__init__(self)
        self.num_neurons = num_neurons
        self.shape = (1, num_neurons)
        self.result = np.zeros(self.shape, dtype=np.double)
        self.error = np.zeros(self.shape, dtype=np.double)

class DimensionalLayer(Layer):
    def __init__(self, width, height, depth):
        Layer.__init__(self)
        self.shape = (height, width, depth)
        self.result = np.zeros(self.shape, dtype=np.double)
        self.error = np.zeros(self.shape, dtype=np.double)
