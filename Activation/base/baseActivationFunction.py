class BaseActivationFunction:

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __call__(self, x):
        return self.forward(x)