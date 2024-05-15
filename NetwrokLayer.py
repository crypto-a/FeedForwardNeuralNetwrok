import numpy as np


class NetworkLayer:
    nodes = []

    def __init__(self):
        self.nodes = []

    def add_perceptron(self, perceptron):
        self.nodes.append(perceptron)

    def forward(self, x):
        return [perceptron.forward(x) for perceptron in self.nodes]

