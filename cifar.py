import datetime
import math
import pickle
import random

import numpy as np

from nano.connection.activation import ReLU, Softmax

from nano.connection.dimensional_to_dimensional import Convolution, MaxPooling, AveragePooling
from nano.connection.dimensional_to_linear import Projection
from nano.connection.linear_to_linear import FullyConnected
from nano.layer import LinearLayer, DimensionalLayer
import nano.network
import nano.trainer

'''
# 80 seconds net - https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg
convnet = nano.network.Network('input', 'output_a')

convnet.add_layer('input', DimensionalLayer(3, 32, 32))
convnet.add_layer('conv1', DimensionalLayer(32, 32, 32))
convnet.add_layer('conv1_a', DimensionalLayer(32, 32, 32))
convnet.add_layer('pool_1', DimensionalLayer(32, 16, 16))

convnet.add_layer('conv2', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv2_a', DimensionalLayer(32, 16, 16))
convnet.add_layer('pool_2', DimensionalLayer(32, 8, 8))

convnet.add_layer('conv3', DimensionalLayer(64, 8, 8))
convnet.add_layer('conv3_a', DimensionalLayer(64, 8, 8))
convnet.add_layer('pool_3', DimensionalLayer(64, 4, 4))

convnet.add_layer('linear1', LinearLayer(64))
convnet.add_layer('linear1_a', LinearLayer(64))
convnet.add_layer('output', LinearLayer(10))
convnet.add_layer('output_a', LinearLayer(10))

convnet.add_connection('input', 'conv1', Convolution(5, 5, 2))
convnet.add_connection('conv1', 'conv1_a', ReLU())
convnet.add_connection('conv1_a', 'pool_1', MaxPooling(2, 2))

convnet.add_connection('pool_1', 'conv2', Convolution(5, 5, 2))
convnet.add_connection('conv2', 'conv2_a', ReLU())
convnet.add_connection('conv2_a', 'pool_2', AveragePooling(2, 2))

convnet.add_connection('pool_2', 'conv3', Convolution(5, 5, 2))
convnet.add_connection('conv3', 'conv3_a', ReLU())
convnet.add_connection('conv3_a', 'pool_3', MaxPooling(2, 2))

convnet.add_connection('pool_3', 'linear1', Projection())
convnet.add_connection('linear1', 'linear1_a', ReLU())
convnet.add_connection('linear1_a', 'output', FullyConnected())
convnet.add_connection('output', 'output_a', Softmax())
'''

# our own definition
convnet = nano.network.Network('input', 'outputA')

convnet.add_layer('input', DimensionalLayer(3, 32, 32))
convnet.add_layer('conv1', DimensionalLayer(16, 32, 32))
convnet.add_layer('conv1A', DimensionalLayer(16, 32, 32))
convnet.add_layer('conv2', DimensionalLayer(16, 32, 32))
convnet.add_layer('conv2A', DimensionalLayer(16, 32, 32))
convnet.add_layer('pool1', DimensionalLayer(16, 16, 16))
convnet.add_layer('conv3', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv3A', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv4', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv4A', DimensionalLayer(32, 16, 16))
convnet.add_layer('pool2', DimensionalLayer(32, 8, 8))
convnet.add_layer('fc1', LinearLayer(32))
convnet.add_layer('fc1A', LinearLayer(32))
convnet.add_layer('output', LinearLayer(10))
convnet.add_layer('outputA', LinearLayer(10))

convnet.add_connection('input', 'conv1', Convolution(3, 3, padding=1))
convnet.add_connection('conv1', 'conv1A', ReLU())
convnet.add_connection('conv1A', 'conv2', Convolution(3, 3, padding=1))
convnet.add_connection('conv2', 'conv2A', ReLU())
convnet.add_connection('conv2A', 'pool1', MaxPooling(2, stride=2))
convnet.add_connection('pool1', 'conv3', Convolution(3, 3, padding=1))
convnet.add_connection('conv3', 'conv3A', ReLU())
convnet.add_connection('conv3A', 'conv4', Convolution(3, 3, padding=1))
convnet.add_connection('conv4', 'conv4A', ReLU())
convnet.add_connection('conv4A', 'pool2', MaxPooling(2, stride=2))
convnet.add_connection('pool2', 'fc1', Projection())
convnet.add_connection('fc1', 'fc1A', ReLU())
convnet.add_connection('fc1A', 'output', FullyConnected())
convnet.add_connection('output', 'outputA', Softmax())


# global hyperparameter
decay_rate = 1e-3
num_validate = 100
num_tick = 500


# read data batch
train_set = []
valid_set = []

for c in range(1, 6):
    tset = open('cifar10/data_batch_%d' % c, 'rb')
    rawdata = pickle.load(tset, encoding='bytes')

    for i in range(0, num_tick):
        img = rawdata[b'data'][i]
        ans = rawdata[b'labels'][i]

        img = img.reshape((3, 32, 32)) / 255
        ansvec = np.zeros(10)
        ansvec[ans] = 1

        if c == 5:
            valid_set.append((img, ansvec))
        else:
            train_set.append((img, ansvec))

    tset.close()


# define trainer
# Nesterov + L2
class MyTrainer(nano.trainer.Trainer):
    def train(self, train_set, epoch, decay_epoch, **kwargs):
        init_func = self.init_func(**kwargs)
        train_func = self.train_func(**kwargs)

        # initialize results
        self.network.learn_prepare(init_func)

        # learn errors
        for current_epoch in range(1, epoch+1):
            random.shuffle(train_set)

            data_loss = 0
            correct = 0
            reg_loss = 0

            counter = 0
            for input_data, output_data in train_set:
                # forward calculation
                result = self.network.calculate(input_data)

                # error propagation
                self.network.learn_error(result - output_data, train_func)

                # data loss
                output_index = np.argmax(output_data)
                result_index = np.argmax(result)
                data_loss += -math.log(result[0][output_index])
                if output_index == result_index:
                    correct += 1

                counter += 1
                if counter % num_tick == 0:  # mini batch

                    # regularization loss
                    for name, layer in self.network.layers.items():
                        for connection in layer.connections:
                            if len(connection.weight) > 0:
                                reg_loss += 0.5 * decay_rate * np.sum(connection.weight[0] * connection.weight[0])

                    if kwargs["batch_func"]:
                        kwargs["batch_func"](current_epoch, counter // num_tick, data_loss / num_tick, reg_loss, correct / num_tick, self.network)

                    data_loss = 0
                    correct = 0
                    reg_loss = 0

            if current_epoch % decay_epoch == decay_epoch - 1:
                kwargs['learning_rate'] *= 0.5
                train_func = self.train_func(**kwargs)

    def init_func(self, **kwargs):
        def init_connection(connection):
            connection.train_prev = []
            connection.train_v = []
            for weight in connection.weight:
                connection.train_prev.append(np.zeros(weight.shape))
                connection.train_v.append(np.zeros(weight.shape))
        return init_connection

    def train_func(self, *, weight_decay, momentum_rate, learning_rate, **kwargs):
        def train_connection(connection):
            for i in range(len(connection.weight)):
                # L2 Regularization
                connection.dweight[i] += weight_decay * connection.weight[i]

                # Nesterov
                connection.train_prev[i][:] = connection.train_v[i]
                connection.train_v[i][:] = -learning_rate * connection.dweight[i] + momentum_rate * connection.train_v[i]
                connection.weight[i] += -momentum_rate * connection.train_prev[i] + (1 + momentum_rate) * connection.train_v[i]
        return train_connection


# define loss function
def my_loss(network, data_set):
    # calculate data loss
    num_data = len(data_set)
    data_loss = 0
    correct = 0
    for input_data, output_data in data_set:
        result = network.calculate(input_data)
        output_index = np.argmax(output_data)
        result_index = np.argmax(result)
        data_loss += -math.log(result[0][output_index])
        if output_index == result_index:
            correct += 1
    data_loss /= num_data

    # calculate regularization loss
    reg_loss = 0
    for name, layer in network.layers.items():
        for connection in layer.connections:
            if len(connection.weight) > 0:
                reg_loss += 0.5 * decay_rate * np.sum(connection.weight[0] * connection.weight[0])

    # (data loss, regularization loss, success_rate)
    return (data_loss, reg_loss, correct / num_data)

# try to load backup
try:
    convnet.load_weight('cifar.npz')
    print('loaded')
except Exception as e:
    print(e)
    print('load failed')




def log(file, str):
    print(str)
    file.write(str + '\n')


def log_result(epoch, batch, train_dloss, train_rloss, train_rate, network):
    log_file = open('cifar.log', 'a')

    convnet.save_weight('cifar.npz')
    log(log_file, str(datetime.datetime.now()))
    log(log_file, 'Epoch %d, Batch %d' % (epoch, batch))

    random.shuffle(valid_set)
    dloss, rloss, rate = my_loss(network, valid_set[:num_validate])
    log(log_file, 'Training %d sample\t dloss %10g rloss %10g rate %10g' % (num_tick, train_dloss, train_rloss, train_rate))
    log(log_file, 'Validate %d sample\t dloss %10g rloss %10g rate %10g' % (num_validate, dloss, rloss, rate))
    log(log_file, '')

    log_file.close()

# train start
trainer = MyTrainer(convnet)
trainer.train(train_set, epoch=10000, weight_decay=decay_rate, momentum_rate=0.9, learning_rate=0.01, decay_epoch=5, batch_func=log_result)

