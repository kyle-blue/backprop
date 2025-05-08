from enum import Enum
from functools import reduce
from typing import Set


# For simplicity (so I don't have to repeat implementation), I have only defined necessary operations
# E.g. SUB can be defined as op1 + (-op2) and DIV can be defined as op1 * op2**-1
class Operation(Enum):
    NONE = ""
    ADD = "+"
    TANH = "tanh"
    EXP = "**"
    MUL = "*"


class Value:
    def __init__(self, data: float, label="", _children=(), _op=Operation.NONE):
        self.data = data
        self.label = label
        self._children: Set[Value] = set(_children)
        self._op = _op
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other: "Value"):
        value = Value(
            self.data + other.data, _children=(self, other), _op=Operation.ADD
        )
        return value

    def __mul__(self, other: "Value"):
        value = Value(
            self.data * other.data, _children=(self, other), _op=Operation.MUL
        )
        return value

    # Calculate the gradients of children with respect to this Value
    def _backward(self):
        for child in self._children:
            match self._op:
                case Operation.ADD:
                    child.grad = self.grad

                case Operation.MUL:
                    others = self._children - set([child])
                    others_data = map(lambda x: x.data, others)
                    child.grad = self.grad * reduce(lambda x, y: x * y, others_data)

                case Operation.TANH:
                    pass

                case Operation.EXP:
                    pass
