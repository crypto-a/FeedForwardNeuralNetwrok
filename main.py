import numpy as np
from Perceptron.perceptron import Perceptron

if __name__ == '__main__':
    # Create a perceptron with 3 features and ReLU activation function
    perceptron = Perceptron(num_features=3, activation_function_name="relu")

    # Forward pass
    x = np.array([1, 2, 3])
    output = perceptron.forward(x)
    print(f"Output: {output}")