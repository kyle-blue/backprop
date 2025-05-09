from typing import List
from mlp import LinearLayer, Module, ReLU, Tanh
from value import Value
from graph import plot_graph


class Model(Module):
    def __init__(self):
        self.l1 = LinearLayer(4, 20)
        self.l2 = LinearLayer(20, 20)
        self.l3 = LinearLayer(20, 1)
        self.relu = ReLU()
        self.tanh = Tanh()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters() + self.l3.parameters()

    def forward(self, x: List[Value]):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.tanh(out)
        return out

    def __call__(self, x: List[Value]):
        return self.forward(x)


def main():
    model = Model()

    loss = Value(0.0)
    plot_graph(loss)


if __name__ == "__main__":
    x = [
        [1.123, 0.01123, -12.123, 3.0],
        [-3.0, -5.1, 0.12, 1.0],
        [-13, 1.0, 10.534, 123.12],
        [1.0, 1.0, 1.0, 1.0],
    ]
    y = [1, 0, 0, 1]
    main()
