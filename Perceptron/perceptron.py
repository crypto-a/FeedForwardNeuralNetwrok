import numpy as np
from ActivationFunctions.base.base import BaseActivationFunction
from ActivationFunctions.Relu import Relu


class Perceptron:
    weights: np.ndarray
    bias: float
    activation_function: BaseActivationFunction

    def __init__(self, num_features: int, activation_function_name: str):
        self.weights = np.random.rand(num_features)
        self.bias = 0.0

        # Initialize the activation function
        match activation_function_name:
            case "relu":
                self.activation_function = Relu()
            # case "sigmoid":
            #     self.activation_function = Sigmoid()
            case _:
                raise ValueError(f"Activation function {activation_function_name} not supported.")

    def forward(self, x: np.ndarray) -> float:
        return self.activation_function.forward(np.dot(x, self.weights) + self.bias)






