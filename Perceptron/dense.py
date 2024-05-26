import numpy as np
from .Activation.base.baseActivationFunction import BaseActivationFunction
from .Activation.relu import ReLU
from .Activation.softmax import Softmax

class Dense:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        activation_cases = {
            'relu': ReLU,
            'softmax': Softmax
        }

        if activation not in activation_cases:
            raise ValueError(f"Activation function {activation} is not available")

        self.activation = activation_cases[activation]()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output_preactivation = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation.forward(self.output_preactivation)
        return self.output

    def backward(self, dvalues: np.ndarray):
        self.dactivation = self.activation.backward(dvalues, self.output)
        self.dweights = np.dot(self.inputs.T, self.dactivation)
        self.dbiases = np.sum(self.dactivation, axis=0, keepdims=True)
        self.dinputs = np.dot(self.dactivation, self.weights.T)

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbiases