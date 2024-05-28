from Netwrok import FeedForward

import numpy as np


# Create instances of the FeedForward network
ffnn = FeedForward(n_inputs=3)
ffnn.add_layer(layer_type='dense', n_neurons=5, activation='relu')
ffnn.add_layer(layer_type='dense', n_neurons=2, activation='softmax')
ffnn.compile(loss='CCE', optimizer='ADAM')

# Training data
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[0, 1], [1, 0]])

# Train the network
ffnn.train(X, y, epochs=100, learning_rate=0.001)
