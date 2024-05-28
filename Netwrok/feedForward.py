from Base import BaseOptimizer
from Layers import Dense
from Loss import MeanSquaredError, CategoricalCrossEntropy
from Optimizers import SGD, Adam

import numpy as np


class FeedForward(BaseOptimizer):
    """
    FeedForward neural network.

    >>> from Optimizers import Adam
    >>> from Loss import CategoricalCrossEntropy
    >>> ffnn = FeedForward(n_inputs=3)
    >>> ffnn.add_layer(layer_type='dense', n_neurons=5, activation='relu')
    >>> ffnn.add_layer(layer_type='dense', n_neurons=2, activation='softmax')
    >>> ffnn.compile(loss='CCE', optimizer='ADAM')
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[0, 1], [1, 0]])
    >>> ffnn.train(X, y, epochs=100, learning_rate=0.001)
    Epoch 0 - Loss: ...
    >>> predictions = ffnn.predict(X)
    >>> predictions.shape
    (2, 2)
    """

    def __init__(self, n_inputs):
        """
        Initialize the FeedForward network.

        :param n_inputs: Number of input features.
        """
        self.input_dimensions = [n_inputs]
        self.layers = []
        self.is_compiled = False

    def add_layer(self, layer_type: str, n_neurons: int, activation: str):
        """
        Add a layer to the network.

        :param layer_type: Type of the layer ('dense', etc.).
        :param n_neurons: Number of neurons in the layer.
        :param activation: Activation function for the layer.

        :raises ValueError: If the network has already been compiled or the layer type is not available.
        """
        if self.is_compiled:
            raise ValueError("Cannot add layer after compilation")

        layer_types = {
            'dense': Dense,
            # Add other layer types here
        }

        if layer_type not in layer_types:
            raise ValueError(f"Layer type {layer_type} is not available")

        n_inputs = self.input_dimensions[-1]
        self.layers.append(layer_types[layer_type](n_inputs, n_neurons, activation))
        self.input_dimensions.append(n_neurons)

    def compile(self, loss: str = 'MSE', optimizer='ADAM'):
        """
        Compile the network with the specified loss function and optimizer.

        :param loss: Loss function to use ('MSE', 'CCE', etc.).
        :param optimizer: Optimizer to use ('SGD', 'ADAM', etc.).

        :raises ValueError: If the loss function or optimizer is not available.
        """
        loss_cases = {
            'MSE': MeanSquaredError,
            'CCE': CategoricalCrossEntropy,
            # Add other loss functions here
        }

        optimizer_cases = {
            'SGD': SGD,
            'ADAM': Adam,
            # Add other optimizers here
        }

        if loss not in loss_cases:
            raise ValueError(f"Loss function {loss} is not available")

        if optimizer not in optimizer_cases:
            raise ValueError(f"Optimizer {optimizer} is not available")

        self.loss = loss_cases[loss]()
        self.optimizer = optimizer_cases[optimizer]()

        self.is_compiled = True

    def forward(self, inputs):
        """
        Perform a forward pass through the network.

        :param inputs: Input data.
        :return: Output of the network.
        :raises ValueError: If the network is not compiled.
        """
        if not self.is_compiled:
            raise ValueError("Network is not compiled")

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, d_values):
        """
        Perform a backward pass through the network.

        :param d_values: Gradient of the loss with respect to the network output.
        """
        for layer in reversed(self.layers):
            d_values = layer.backward(d_values)

    def train(self, X, y, epochs, learning_rate):
        """
        Train the network.

        :param X: Training data.
        :param y: Training labels.
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.optimizer.learning_rate = learning_rate
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.loss.forward(predictions, y)

            if epoch % 100 == 0:
                print(f'Epoch {epoch} - Loss: {loss}')

            d_values = self.loss.backward(predictions, y)
            self.backward(d_values)
            self.update_parameters()

    def predict(self, X):
        """
        Make predictions with the network.

        :param X: Input data.
        :return: Network predictions.
        """
        return self.forward(X)

    def update_parameters(self):
        """
        Update the parameters of the network using the optimizer.
        """
        for layer in self.layers:
            self.optimizer.update(layer.weights, layer.d_weights)
            if layer.bias is not None and layer.d_bias is not None:
                self.optimizer.update(layer.bias, layer.d_bias)
