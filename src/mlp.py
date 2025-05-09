from typing import List
import random
from value import Value


class Module:
    def parameters(self) -> List[Value]:  # Implement this
        return []

    def zero_grad(self):
        params = self.parameters()
        for param in params:
            param.grad = 0


class LinearNeuron(Module):
    def __init__(self, n_inputs: int):
        # Here we use He/Kaiming initialisation for ReLU
        # We would use Xaviar/Glorot initialisation for sigmoid or tanh activation
        # NOTE: Maybe look into this a little more to figure out where these values came from
        a = (6 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-a, a)) for _ in range(n_inputs)]
        self.bias = Value(0)  # This is just learned

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron(n_inputs={len(self.weights)})"

    def forward(self, x: List[Value]):
        out = Value(0)
        out += sum([xi * wi for xi, wi in zip(x, self.weights)] + [self.bias])
        return out

    def __call__(self, x: List[Value]):
        return self.forward(x)


class LinearLayer(Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.neurons = [LinearNeuron(n_inputs) for _ in range(n_outputs)]

    def parameters(self):
        out = []
        for n in self.neurons:
            out += n.parameters()
        return out

    def __repr__(self):
        return (
            f"LinearLayer(n_inputs = {self.n_inputs}, n_outputs = {len(self.neurons)})"
        )

    def forward(self, x: List[Value]):
        out = [n.forward(x) for n in self.neurons]
        return out

    def __call__(self, x: List[Value]):
        return self.forward(x)


class ReLU(Module):
    def forward(self, x: List[Value]):
        return [a.relu() for a in x]

    def __repr__(self):
        return "ReLU()"

    def __call__(self, x: List[Value]):
        return self.forward(x)


class Tanh(Module):
    def forward(self, x: List[Value]):
        return [a.tanh() for a in x]

    def __repr__(self):
        return "Tanh()"

    def __call__(self, x: List[Value]):
        return self.forward(x)
