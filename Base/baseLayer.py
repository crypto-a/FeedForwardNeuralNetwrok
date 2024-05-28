




class BaseLayer:

    def forward(self, input):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError