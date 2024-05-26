import numpy as np

np.random.seed(0)


class LayerDense:
    weights = np.ndarray
    biases = np.ndarray
    output = np.ndarray

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # change direction of matrix to prevent transposing
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
