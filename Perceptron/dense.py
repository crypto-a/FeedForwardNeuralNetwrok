import numpy as np
from Activation.base.baseActivationFunction import BaseActivationFunction
from Activation.relu import ReLU
from Activation.softmax import Softmax


class Dense:
    inputs: np.ndarray

    wights: np.ndarray
    bias: float
    activation: BaseActivationFunction

    output_preactivation: np.ndarray
    output: np.ndarray

    def __init__(self, n_inputs: int, activation: str):
        """
        This class will create a dense perceptron with n_inputs and activation function
        :param n_inputs:
        :param activation:
        """
        self.weights = 0.10 * np.random.randn(n_inputs, 1)
        self.bias = 0

        activation_cases = {
            'relu': ReLU,
            'softmax': Softmax
        }

        # check if the activation function is available
        if activation not in activation_cases.keys():
            raise ValueError(f"Activation function {activation} is not available")

        self.activation = activation_cases.get(activation, ReLU)()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        This function will forward the inputs to the perceptron
        :param inputs:
        :return:
        """
        self.inputs = inputs
        self.output_preactivation = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output
