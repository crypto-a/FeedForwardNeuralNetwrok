import numpy as np
from Perceptron.dense import Dense


class Layer:
    layer: list[Dense]

    layer_output: np.ndarray

    def __init__(self, n_inputs: int, n_neurons: int, activation: str):
        self.layer = [Dense(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, inputs):
        self.layer_output = np.array([layer.forward(inputs) for layer in self.layer])
        return self.layer_output





