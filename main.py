from layer import LayerDense
from Perceptron.Activation.relu import ReLU
from Perceptron.Activation.softmax import Softmax


if __name__ == '__main__':
    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(X)
    layer1_activation = ReLU()
    layer1_activation.forward(layer1.output)
    layer2.forward(layer1_activation.output)
    layer2_activation = Softmax()
    layer2_activation.forward(layer2.output)

    print(layer2_activation.output)