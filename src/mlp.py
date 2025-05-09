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


class Neuron(Module):
    def __init__(self, n_inputs: int, is_linear=False):
        # Here we use He/Kaiming initialisation for ReLU
        # We would use Xaviar/Glorot initialisation for sigmoid or tanh activation
        # NOTE: Maybe look into this a little more to figure out where these values came from
        a = (6 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-a, a)) for _ in range(n_inputs)]
        self.bias = Value(0)  # This is just learned
        self.is_linear = is_linear

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron(n_inputs={len(self.weights)}, is_linear={self.is_linear})"

    def forward(self, x: List[Value]):
        out = Value(0)
        out += sum([xi * wi for xi, wi in zip(x, self.weights)] + [self.bias])
        if not self.is_linear:
            out = out.relu()
        return out

    def __call__(self, x: List[Value]):
        return self.forward(x)
