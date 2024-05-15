from NetwrokLayer import NetworkLayer
from .Perceptron.perceptron import Perceptron


class ForwardFeedNeuralNetwork:
    layers: list[NetworkLayer]

    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_perceptrons, activation_function_name):
        layer = NetworkLayer()
        for _ in range(number_of_perceptrons):
            layer.add_perceptron(Perceptron(len(self.layers[-1].nodes), activation_function_name))
        self.layers.append(layer)
