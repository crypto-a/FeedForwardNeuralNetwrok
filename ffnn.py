from layer import Layer

class ForwardFeed:
    Layers: list[Layer]
    Input_dimensions: list[int]

    is_compiled: bool

    def __init__(self, n_inputs):
        self.Input_dimensions = [n_inputs]
        self.Layers = []
        self.is_compiled = False

    def add_layer(self, n_neurons: int, activation: str):
        """
        This function will add a layer to the network
        :param n_neurons:
        :param activation:
        :return:
        """
        if self.is_compiled:
            raise ValueError("Cannot add layer after compilation")

        n_inputs = self.Input_dimensions[-1]
        self.Layers.append(Layer(n_inputs, n_neurons, activation))
        self.Input_dimensions.append(n_neurons)

    def compile(self):
        """
        This function will compile the network
        :return:
        """
        self.is_compiled = True

    def forward(self, inputs):
        """
        This function will forward the inputs to the network
        :param inputs:
        :return:
        """
        if not self.is_compiled:
            raise ValueError("Network is not compiled")

        for layer in self.Layers:
            inputs = layer.forward(inputs)

        return inputs