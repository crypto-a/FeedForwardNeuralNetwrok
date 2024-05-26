from ffnn import ForwardFeed
import numpy as np

from Perceptron.dense import Dense
from layer import Layer

if __name__ == "__main__":
    # Create the neural network
    ffnn = ForwardFeed(n_inputs=3)

    # Add layers
    ffnn.add_layer(n_neurons=4, activation='relu')
    ffnn.add_layer(n_neurons=1, activation='softmax')

    # Compile the network
    ffnn.compile()

    # Training data
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    ffnn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the neural network
    predictions = ffnn.predict(X)
    print("Predicted Output:")
    print(predictions)

    # layer1 = Layer(n_inputs=3, n_neurons=4, activation='relu')
    # layer2 = Layer(n_inputs=4, n_neurons=5, activation='relu')
    # layer3 = Layer(n_inputs=5, n_neurons=2, activation='softmax')
    #
    # inputs = np.array([[0, 0, 1],
    #                    [0, 1, 1],
    #                    [1, 0, 1],
    #                    [1, 1, 1]])
    #
    # layer1_output = layer1.forward(inputs)
    # print(layer1_output)
    # layer2_output = layer2.forward(layer1_output)
    # print(layer2_output)
    # layer3_output = layer3.forward(layer2_output)
    # print(layer3_output)



