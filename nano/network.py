import numpy as np

from nano.connection.activation import AbsActivation


class NetworkError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class Network:
    def __init__(self, input_layer_name, output_layer_name):
        self.input_layer_name = input_layer_name
        self.output_layer_name = output_layer_name

        self.layers = dict()

        self.fixed = False
        self.topologic_order = None

    def add_layer(self, name, layer):
        if self.fixed:
            raise NetworkError("Network is fixed")

        if layer.added:
            raise NetworkError("The layer is already used")

        layer.added = True
        self.layers[name] = layer
        self.fixed = False

    def add_connection(self, before, after, connection):
        if before in self.layers:
            before_layer = self.layers[before]
        else:
            raise NetworkError("Network not found")

        if after in self.layers:
            after_layer = self.layers[after]
        else:
            raise NetworkError("Network not found")

        connection.connect(before_layer, after_layer)
        self.fixed = False

    def fix(self):
        if not self.fixed:
            for name, layer in self.layers.items():
                layer.pre_layer_connect = 0

                # check reachability
                if name != self.input_layer_name and layer.pre_layer_count == 0:
                    raise NetworkError("Unreachable Layer Exists")

                # check stuck
                if name != self.output_layer_name and len(layer.connections) == 0:
                    raise NetworkError("Network Stuck")

            # topology sort
            queue = []
            self.topologic_order = []

            input_layer = self.layers[self.input_layer_name]
            queue.append(input_layer)

            while len(queue) > 0:
                now_layer = queue.pop()
                self.topologic_order.append(now_layer)
                for connection in now_layer.connections:
                    next_layer = connection.after_layer
                    next_layer.pre_layer_connect += 1
                    if next_layer.pre_layer_count == next_layer.pre_layer_connect:
                        queue.append(next_layer)

            self.fixed = True

    def calculate(self, input_data):
        self.fix()

        # initialize result
        for name, layer in self.layers.items():
            layer.result.fill(0)

        # run neural net
        np.copyto(self.layers[self.input_layer_name].result, input_data)
        for layer in self.topologic_order:
            for connection in layer.connections:
                connection.forward()

        return self.layers[self.output_layer_name].result

    def learn_prepare(self, init_func):
        self.fix()

        # initialize result
        for name, layer in self.layers.items():
            for connection in layer.connections:
                init_func(connection)

    def learn_error(self, error_data, train_func):
        # back propagation
        for name, layer in self.layers.items():
            layer.error.fill(0)

        # run neural net
        np.copyto(self.layers[self.output_layer_name].error, error_data)
        for layer in reversed(self.topologic_order):
            for connection in layer.connections:
                connection.backward()
                train_func(connection)
            ''''print(layer.error)
            print(layer.result)
            input()'''
