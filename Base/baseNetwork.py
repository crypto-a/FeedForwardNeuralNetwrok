

class BaseNetwork:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError

    def update(self, learning_rate):
        pass

    def __call__(self, x):
        return self.forward(x)