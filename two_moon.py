import numpy as np

from nano.connection.activation import ReLU, Sigmoid
from nano.connection.linear_to_linear import FullyConnected
from nano.layer import LinearLayer
import nano.network
import nano.trainer


# initialize network
mlp = nano.network.Network('input', 'output_a')

mlp.add_layer('input', LinearLayer(2))

mlp.add_layer('hidden1', LinearLayer(50))
mlp.add_layer('hidden1_a', LinearLayer(50))

mlp.add_connection('input', 'hidden1', FullyConnected())
mlp.add_connection('hidden1', 'hidden1_a', Sigmoid())

mlp.add_layer('hidden2', LinearLayer(30))
mlp.add_layer('hidden2_a', LinearLayer(30))

mlp.add_connection('hidden1_a', 'hidden2', FullyConnected())
mlp.add_connection('hidden2', 'hidden2_a', Sigmoid())

mlp.add_layer('output', LinearLayer(1))
mlp.add_layer('output_a', LinearLayer(1))

mlp.add_connection('hidden2_a', 'output', FullyConnected())
mlp.add_connection('output', 'output_a', Sigmoid())

# parse input data
data_file = open('two_moon.txt', 'r')

line_num = 0

train_set = []
validation_set = []
test_set = []

for line in data_file.readlines():
    ix, iy, oc = map(float, line.split())

    if line_num < 100:
        train_set.append((np.array([[ix, iy]]), np.array([[oc]])))
    elif line_num < 160:
        validation_set.append((np.array([[ix, iy]]), np.array([[oc]])))
    else:
        test_set.append((np.array([[ix, iy]]), np.array([[oc]])))

    line_num += 1

data_file.close()

# train the network
def log_result(current_epoch, network):
    train_error = nano.trainer.rms(network, train_set)
    validation_error = nano.trainer.rms(network, validation_set)
    print('EPOCH: %d - train error %g / validation error %g' % (current_epoch, train_error, validation_error))

sgd = nano.trainer.SGD(mlp)
sgd.train(train_set, epoch=1000, learning_rate=0.002, epoch_func=log_result)
