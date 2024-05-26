from ffnn import ForwardFeed
import numpy as np

from Perceptron.dense import Dense

if __name__ == '__main__':

    X = np.array([1, 2, 3, 2.5])

    ffnn = ForwardFeed(4)

    ffnn.add_layer(3, 'relu')
    ffnn.add_layer(2, 'relu')
    ffnn.add_layer(1, 'relu')

    ffnn.compile()


    print(ffnn.forward(X))




