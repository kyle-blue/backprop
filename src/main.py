from typing import List, Union
import typing
from mlp import LinearLayer, Module, ReLU, Tanh
from value import Value
from graph import plot_graph


class Model(Module):
    def __init__(self):
        self.l1 = LinearLayer(4, 6)
        self.l2 = LinearLayer(6, 6)
        self.l3 = LinearLayer(6, 6)
        self.l4 = LinearLayer(6, 1)
        self.relu = ReLU()
        self.tanh = Tanh()

    def parameters(self) -> List[Value]:
        return (
            self.l1.parameters()
            + self.l2.parameters()
            + self.l3.parameters()
            + self.l4.parameters()
        )

    def forward(self, x: List[Value]):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.tanh(out)
        out = self.l4(out)
        out = self.tanh(out)
        return out

    def __call__(self, x: List[Value]):
        return self.forward(x)


def mse(preds: List[Value], ys: Union[List[int], List[float]]) -> Value:
    se = [(y - p) ** 2 for p, y in zip(preds, ys)]
    return typing.cast(Value, sum(se) / len(se))


def main():
    model = Model()

    xs = [
        [1.123, 0.01123, -12.123, 3.0],
        [-3.0, -5.1, 0.12, 1.0],
        [-13, 1.0, 10.534, 123.12],
        [1.0, 1.0, 1.0, 1.0],
    ]
    ys = [1, 0, 0, 1]

    training_iters = 30
    loss = Value(0)
    preds: List[Value] = []
    learning_rate = 0.2
    for i in range(training_iters):
        model.zero_grad()
        preds = [model(x)[0] for x in xs]
        loss = mse(preds, ys)
        print(f"{i} --- loss: {loss.data:.4f}")
        loss.backward()

        # Step in direction of gradients
        params = model.parameters()
        for param in params:
            # If gradient is positive, we need it to go down, otherwise up. So we must subtract
            param.data -= learning_rate * param.grad

    print(f"Final guesses: {[p.data for p in preds]}")
    plot_graph(loss, view=True)


if __name__ == "__main__":
    main()
