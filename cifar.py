import pickle

import numpy as np

from nano.connection.activation import ReLU, Sigmoid, Softmax
from nano.connection.dimensional_to_dimensional import Convolution, MaxPooling
from nano.connection.dimensional_to_linear import Projection
from nano.connection.linear_to_linear import FullyConnected
from nano.layer import LinearLayer, DimensionalLayer
import nano.network
import nano.trainer

convnet = nano.network.Network('input', 'outputA')

convnet.add_layer('input', DimensionalLayer(3, 32, 32))
convnet.add_layer('conv1', DimensionalLayer(32, 32, 32))
convnet.add_layer('conv1A', DimensionalLayer(32, 32, 32))
convnet.add_layer('pool1', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv2', DimensionalLayer(32, 16, 16))
convnet.add_layer('conv2A', DimensionalLayer(32, 16, 16))
convnet.add_layer('pool2', DimensionalLayer(32, 8, 8))
convnet.add_layer('conv3', DimensionalLayer(64, 8, 8))
convnet.add_layer('conv3A', DimensionalLayer(64, 8, 8))
convnet.add_layer('mlp1', LinearLayer(64))
convnet.add_layer('mlp1A', LinearLayer(64))
convnet.add_layer('output', LinearLayer(10))
convnet.add_layer('outputA', LinearLayer(10))

convnet.add_connection('input', 'conv1', Convolution(5, 5, padding=2))
convnet.add_connection('conv1', 'conv1A', ReLU())
convnet.add_connection('conv1A', 'pool1', MaxPooling(2, stride=2))
convnet.add_connection('pool1', 'conv2', Convolution(5, 5, padding=2))
convnet.add_connection('conv2', 'conv2A', ReLU())
convnet.add_connection('conv2A', 'pool2', MaxPooling(2, stride=2))
convnet.add_connection('pool2', 'conv3', Convolution(5, 5, padding=2))
convnet.add_connection('conv3', 'conv3A', ReLU())
convnet.add_connection('conv3A', 'mlp1', Projection())
convnet.add_connection('mlp1', 'mlp1A', ReLU())
convnet.add_connection('mlp1A', 'output', FullyConnected())
convnet.add_connection('output', 'outputA', Softmax())

# change this line to your data path
tset = open('cifar10/data_batch_1', 'rb')
rawdata = pickle.load(tset, encoding='bytes')

train_set = []

'''
this is just an example for testing!
you must change the data loading section to fully train the dataset
'''
for i in range(0, 5):
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

moemntum = nano.trainer.SGDMomentum(convnet)
moemntum.train(train_set, epoch=500, momentum_rate=0.9, learning_rate=0.001, epoch_func=log_result)
