from abc import ABCMeta, abstractmethod

import numpy as np


def rms(network, data_set):
    errors = np.empty((network.layers[network.output_layer_name].shape[1], len(data_set)))
    c = 0
    for input_data, output_data in data_set:
        result = network.calculate(input_data)
        error = output_data - result

        errors[:, c] = error
        c += 1
    return np.sqrt(np.mean(np.square(errors)))

class Trainer:
    def __init__(self, network):
        self.network = network
        self.network.fix()

    @abstractmethod
    def train(self):
        pass

class EpochBasedTrainer(Trainer, metaclass=ABCMeta):
    '''
    [required]
    network: Network object
    train_set: tuple of (input_vector, output_vector)

    [optional]
    epoch_func: callback function when an epoch is finished
    '''
    def train(self, train_set, epoch, **kwargs):
        init_func = self.init_func(**kwargs)
        train_func = self.train_func(**kwargs)

        # initialize results
        self.network.learn_prepare(init_func)

        # learn errors
        for current_epoch in range(1, epoch+1):
            for input_data, output_data in train_set:
                # forward calculation
                result = self.network.calculate(input_data)

                # error propagation
                self.network.learn_error(output_data - result, train_func)

            if kwargs["epoch_func"]:
                kwargs["epoch_func"](current_epoch, self.network)

    @abstractmethod
    def init_func(self, **kwargs):
        '''
        Return the function which reads weight list and prepares training
        '''
        pass

    @abstractmethod
    def train_func(self, **kwargs):
        '''
        Return the function which reads dweight and prepares training
        '''
        pass

class SGD(EpochBasedTrainer):
    def init_func(self, **kwargs):
        def init_connection(connection):
            pass
        return init_connection

    '''
    [required]
    learning_rate
    '''
    def train_func(self, *, learning_rate, **kwargs):
        def train_connection(connection):
            for i in range(len(connection.weight)):
                connection.weight[i] += learning_rate * connection.dweight[i]
        return train_connection

class SGDMomentum(EpochBasedTrainer):
    def init_func(self, **kwargs):
        def init_connection(connection):
            connection.train_momentum = []
            for weight in connection.weight:
                connection.train_momentum.append(np.zeros(weight.shape))
        return init_connection

    '''
    [required]
    momentum_rate
    learning_rate
    '''
    def train_func(self, *, momentum_rate, learning_rate, **kwargs):
        def train_connection(connection):
            for i in range(len(connection.weight)):
                current_speed = learning_rate * connection.dweight[i] + momentum_rate * connection.train_momentum[i]
                connection.weight[i] += current_speed
                connection.train_momentum[i][:] = current_speed
        return train_connection
