from layer import Layer
import numpy as np
from Loss.meanSquaredError import MeanSquaredError

class ForwardFeed:
    def __init__(self, n_inputs):
        self.Input_dimensions = [n_inputs]
        self.Layers = []
        self.is_compiled = False
        self.loss = MeanSquaredError()

    def add_layer(self, n_neurons: int, activation: str):
        if self.is_compiled:
            raise ValueError("Cannot add layer after compilation")

        n_inputs = self.Input_dimensions[-1]
        self.Layers.append(Layer(n_inputs, n_neurons, activation))
        self.Input_dimensions.append(n_neurons)

    def compile(self):
        self.is_compiled = True

    def forward(self, inputs):
        if not self.is_compiled:
            raise ValueError("Network is not compiled")

        for layer in self.Layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, dvalues, learning_rate):
        for layer in reversed(self.Layers):
            dvalues = layer.backward(dvalues, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.loss.calculate(predictions, y)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch} - Loss: {loss}')

            dvalues = self.loss.backward(predictions, y)
            self.backward(dvalues, learning_rate)
            self.update_parameters(learning_rate)

    def predict(self, X):
        return self.forward(X)

    def update_parameters(self, learning_rate):
        for layer in self.Layers:
            layer.update_parameters(learning_rate)
