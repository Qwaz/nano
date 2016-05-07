import pickle

import numpy as np

from nano.connection.activation import ReLU, Sigmoid, Softmax
from nano.connection.dimensional_to_dimensional import Convolution, MaxPooling, AveragePooling
from nano.connection.dimensional_to_linear import Projection
from nano.connection.linear_to_linear import FullyConnected
from nano.layer import LinearLayer, DimensionalLayer
import nano.network
import nano.trainer

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

# change this line to your data path
tset = open('<path_to_cifar10_dataset>', 'rb')
rawdata = pickle.load(tset, encoding='bytes')

train_set = []

'''
this is just an example for testing!
you must change the data loading section to fully train the dataset
'''
for i in range(0, 20):
	img = rawdata[b'data'][i]
	ans = rawdata[b'labels'][i]

	img = img / 255
	img = img.reshape((3, 32, 32))
	ansvec = np.zeros(10)
	ansvec[ans] = 1

	train_set.append((img, ansvec))

tset.close()
print('loaded')
def log_result(current_epoch, network):
    loss, rate = nano.trainer.softmax(network, train_set)
    print('EPOCH: %d - loss %g, rate %g' % (current_epoch, loss, rate))

momentum = nano.trainer.SGDMomentum(convnet)
momentum.train(train_set, epoch=1000, momentum_rate=0.9, learning_rate=0.0001, epoch_func=log_result)
