import numpy as np
from Perceptron.dense import Dense

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str):
        self.dense_layer = Dense(n_inputs, n_neurons, activation)

    def forward(self, inputs):
        self.layer_output = self.dense_layer.forward(inputs)
        return self.layer_output

    def backward(self, dvalues, learning_rate):
        self.dense_layer.backward(dvalues)
        self.dense_layer.weights -= learning_rate * self.dense_layer.dweights
        self.dense_layer.bias -= learning_rate * self.dense_layer.dbiases
        return self.dense_layer.dinputs

    def update_parameters(self, learning_rate):
        self.dense_layer.update_parameters(learning_rate)
