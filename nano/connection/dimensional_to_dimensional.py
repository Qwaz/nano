from abc import ABCMeta

import numpy as np

from nano.connection import AbsConnection
from nano.layer import DimensionalLayer
from nano.layer import LinearLayer

class DimensionToDimension(AbsConnection, metaclass=ABCMeta):
	def __init__(self):
		super().__init__()

	def check_layer_type(self, before_layer, after_layer):
		return isinstance(before_layer, DimensionalLayer) and isinstance(after_layer, DimensionalLayer)

class DimensionToLinear(AbsConnection, metaclass=ABCMeta):
	def __init__(self):
		super().__init__()

	def check_layer_type(self, before_layer, after_layer):
		return isinstance(before_layer, DimensionalLayer) and isinstance(after_layer, LinearLayer)

class Convolution(DimensionToDimension):
	def __init__(self, height, width, stride=1):
		super().__init__()
		self.stride = stride
		self.height = height
		self.width = width

	def check_shape(self, before_shape, after_shape):
		if ((before_shape[1] - self.height) // self.stride + 1 == after_shape[1]
			and (before_shape[2] - self.width) // self.stride + 1 == after_shape[2]):
			return True
		else:
			return False

	def prepare_connection(self, before_shape, after_shape):
		self.depth = before_shape[0]
		self.filters = after_shape[0]
		for i in range(0, self.filters):
			self.weight.append(# filters
				2 * np.random.random((self.depth, self.height, self.width)) - 1 
			)
		self.weight.append(# weights
			np.random.random(self.filters)
		)

	def forward(self):
		for i in range(0, self.filters):
			filter_vec = self.weight[i].flatten()
			bias = np.array([self.weight[self.filters][i]])

			calc_matrix = np.zeros((self.height * self.width * self.depth , 1))
			for j in range(0, self.after_layer.shape[1]):
				for k in range(0, self.after_layer.shape[2]):
					temp = np.zeros((1, 1))
					for d in range(0, self.depth):
						layer_values = self.before_layer.result[d, j * self.stride : j * self.stride + self.height, k * self.stride : k * self.stride + self.width]
						layer_values = layer_values.reshape((self.height * self.width , 1))
						temp = np.vstack((temp, layer_values))
					temp = temp[1 : , :]
					calc_matrix = np.hstack((calc_matrix, temp))

			calc_matrix = calc_matrix[:, 1 : ]
			res = np.dot(filter_vec, calc_matrix)
			res = res + np.tile(bias, self.after_layer.shape[1] * self.after_layer.shape[2])
			res = res.reshape((self.after_layer.shape[1], self.after_layer.shape[2]))
			self.after_layer.result[i, :, :] = res
		'''
		for i in range(0, self.filters):
			for j in range(0, self.after_layer.shape[1]):
				for k in range(0, self.after_layer.shape[2]):
					self.after_layer.result[i, k, j] += 
					np.sum(numpy.multiply(self.before_layer.result[: ,self.stride * j : self.stride * j + self.height, self.stride * k : self.stride * k + self.width], self.weight[i]))
					+ np.weight[self.filters][i]
		'''
	def backward(self):
		'''
		backprop currently supports only strides of size 1!!!
		'''
		back_error_prop = np.zeros((self.depth, self.before_layer.shape[1], self.before_layer.shape[2]))
		for i in range(0, self.filters):
			temp_weights = self.after_layer.error[i, :, :]
			for j in range(0, self.height - 1):
				temp_weights = np.insert(temp_weights, self.after_layer.shape[1], 0, axis=0)
			for j in range(0, self.height - 1):
				temp_weights = np.insert(temp_weights, 0, 0, axis=0)
			for j in range(0, self.width - 1):
				temp_weights = np.insert(temp_weights, self.after_layer.shape[2], 0, axis=1)
			for j in range(0, self.width - 1):
				temp_weights = np.insert(temp_weights, 0, 0, axis=1)
			# zero padding for size of matrix
			'''
			for j in range(0, self.after_layer.shape[1]):
				for k in range(0, self.stride - 1):
					temp_weights = np.insert(temp_weights, self.height + i * self.stride, 0, axis=0)
			for j in range(0, self.after_layer.shape[2]):
				for k in range(0, self.stride - 1):
					temp_weights = np.insert(temp_weights, self.width + i * self.stride, 0, axis=1)
			#zero padding complete
			'''
			error_matrix = np.zeros((self.height * self.width, 1))
			for j in range(0, self.before_layer.shape[1]):
				for k in range(0, self.before_layer.shape[2]):
					temp = temp_weights[j * self.stride : j * self.stride + self.height , k * self.stride : k * self.stride + self.width]
					temp = temp.reshape(self.height * self.width, 1)
					error_matrix = np.hstack((error_matrix, temp))

			error_matrix = error_matrix[:, 1 : ]
			#create the matrix of errors		

			for j in range(0, self.depth):
				filter_vec = self.weight[i][j, :, :].flatten()
				filter_vec = filter_vec[: : -1]
				back_error = np.dot(filter_vec, error_matrix)
				back_error = back_error.reshape(self.before_layer.shape[1], self.before_layer.shape[2])
				back_error_prop[j, :, :] += back_error


			#reverse the filter vector and propagate the backwards error term
		self.before_layer.error = back_error_prop

		for i in range(0, self.filters):
			filter_vec = self.after_layer.error[i, :, :] # this is the filter for the convolution
			filter_vec = filter_vec.flatten() # we flatten it
			bias_update = filter_vec.sum()
			
			calc_matrix = np.zeros((self.after_layer.shape[1] * self.after_layer.shape[2], 1))
			for d in range(0, self.depth):
				for j in range(0, self.height):
					for k in range(0, self.width):
						temp = np.zeros((1, 1))
						layer_values = self.before_layer.result[d, j * self.stride : j * self.stride + self.after_layer.shape[1], k * self.stride : k * self.stride + self.after_layer.shape[2]]
						layer_values = layer_values.reshape(self.after_layer.shape[1] * self.after_layer.shape[2], 1)
						temp = np.vstack((temp, layer_values))
						temp = temp[1 : , :]
						calc_matrix = np.hstack((calc_matrix, temp))

			calc_matrix = calc_matrix[:, 1 : ]
			error_matrix = np.dot(filter_vec, calc_matrix)
			error_matrix = error_matrix.reshape(self.depth, self.height, self.width)

			np.copyto(self.dweight[i], error_matrix) #update filter
			self.dweight[self.filters][i] = bias_update

class MaxPooling(DimensionToDimension):
	def __init__(self, height, width, stride):
		super().__init__()
		self.height = height
		self.width = width
		self.stride = stride

	def check_shape(self, before_shape, after_shape):
		if (before_shape[0] == after_shape[0]
			and (before_shape[1] - self.height) // self.stride + 1 == after_shape[1]
			and (before_shape[2] - self.width) // self.stride + 1 == after_shape[2]):
			return True
		else:
			return False
	def prepare_connection(self, before_shape, after_shape):
		pass
	def forward(self):
		for i in range(0, self.after_layer.shape[0]):
			for j in range(0, self.after_layer.shape[1]):
				for k in range(0, self.after_layer.shape[2]):
					self.after_layer.result[i, j, k] = (
						np.amax(self.before_layer.result[i, self.stride * j : self.stride * j + self.height, self.stride * k : self.stride * k + self.width]))

	def backward(self):
		self.before_layer.error.fill(0)
		for i in range(0, self.after_layer.shape[0]):
			for j in range(0, self.after_layer.shape[1]):
				for k in range(0, self.after_layer.shape[2]):
					ind = np.unravel_index(np.argmax(self.before_layer.result[i, self.stride * j : self.stride * j + self.height, self.stride * k : self.stride * k + self.width]), (1, self.height, self.width))
					d = ind[0] + i
					h = ind[1] + self.stride * j
					w = ind[2] + self.stride * k
					self.before_layer.error[(d, h, w)] = self.after_layer.error[i, j, k]

class Projection(DimensionToLinear):
	def __init__(self):
		super().__init__()
	def check_shape(self, before_shape, after_shape):
		return True
	def prepare_connection(self, before_shape, after_shape):
		self.weight.append(
			2 * np.random.random((before_shape[0] * before_shape[1] * before_shape[2], after_shape[1])) - 1
		)
		self.weight.append(
			np.random.random(after_shape[1])
		)
	def forward(self):
		input_temp = self.before_layer.result.flatten()
		self.after_layer.result += np.dot(input_temp, self.weight[0]) + self.weight[1]

	def backward(self):

		dw = np.repeat(np.array([self.before_layer.result.flatten()]).T, self.after_layer.shape[1], axis=1)
		dw = np.multiply(dw, self.after_layer.error)
		np.copyto(self.dweight[0], dw)
		np.copyto(self.dweight[1], self.after_layer.error)

		de = np.multiply(self.weight[0], self.after_layer.error)
		de = np.sum(de, axis=1)
		de = np.reshape(de, self.before_layer.shape)
		np.copyto(self.before_layer.error, de)
		