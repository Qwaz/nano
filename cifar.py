import pickle

import numpy as np

from nano.connection.activation import ReLU, Linear, Sigmoid
from nano.connection.dimensional_to_dimensional import Convolution, MaxPooling
from nano.connection.dimensional_to_linear import Projection
from nano.connection.linear_to_linear import FullyConnected
from nano.layer import LinearLayer, DimensionalLayer
import nano.network
import nano.trainer

convnet = nano.network.Network('input', 'output_a')

convnet.add_layer('input', DimensionalLayer(3, 32, 32))
convnet.add_layer('conv_1', DimensionalLayer(2, 31, 31))
convnet.add_layer('conv_1a', DimensionalLayer(2, 31, 31))
convnet.add_layer('pool_1', DimensionalLayer(2, 5, 5))
convnet.add_layer('mlp_in', LinearLayer(20))
convnet.add_layer('mlp_ina', LinearLayer(20))
convnet.add_layer('output', LinearLayer(10))
convnet.add_layer('output_a', LinearLayer(10))

convnet.add_connection('input', 'conv_1', Convolution(2, 2))
convnet.add_connection('conv_1', 'conv_1a', ReLU())
convnet.add_connection('conv_1a', 'pool_1', MaxPooling(6, 6, 6))
convnet.add_connection('pool_1', 'mlp_in', Projection())
convnet.add_connection('mlp_in', 'mlp_ina', ReLU())
convnet.add_connection('mlp_ina', 'output', FullyConnected())
convnet.add_connection('output', 'output_a', Linear())

# change this line to your data path
tset = open('/Users/Admin/Documents/Workspace/postech/CITE505/HW2/nn-assign2/cifar10/data_batch_1', 'rb')
rawdata = pickle.load(tset, encoding='bytes')

train_set = []

'''
this is just an example for testing!
you must change the data loading section to fully train the dataset
'''
for i in range(0, 300):
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
    train_error = nano.trainer.rms(network, train_set)
    print('EPOCH: %d - train error %g' % (current_epoch, train_error))

sgd = nano.trainer.SGD(convnet)
sgd.train(train_set, epoch=100, learning_rate=0.001, epoch_func=log_result)
