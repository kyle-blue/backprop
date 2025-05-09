from enum import Enum
import math
from functools import reduce
from typing import List, Set, Tuple, Union


# For simplicity (so I don't have to repeat implementation), I have only defined necessary operations
# E.g. SUB can be defined as op1 + (-op2) and DIV can be defined as op1 * op2**-1
class Operation(Enum):
    NONE = ""
    ADD = "+"
    TANH = "tanh"
    POW = "**"
    MUL = "*"
    RELU = "relu"
    EXP = "exp"


accepted_types = Union["Value", int, float]


class Value:
    def __init__(
        self,
        data: float,
        label="",
        _children: Tuple["Value", ...] = (),
        _op=Operation.NONE,
    ):
        self.data = data
        self.label = label
        self._children = _children
        self._op = _op
        self.grad = 0.0

    # Calculate the gradients of children with respect to this Value
    # Child grads must accumulate, as a single node can (and most likely will) be used more than once
    def _backward(self):
        for i, child in enumerate(self._children):
            match self._op:
                case Operation.ADD:
                    child.grad += self.grad

                case Operation.MUL:
                    others = [x.data for x in self._children]
                    others.pop(i)
                    others_product = reduce(lambda x, y: x * y, others)
                    child.grad += self.grad * others_product

                case Operation.POW:
                    # Only allowed two operands in this instance
                    assert len(self._children) == 2
                    base = self._children[0]
                    power = self._children[1]
                    if child == base:
                        child.grad += (
                            power.data * base.data ** (power.data - 1) * self.grad
                        )
                    else:  # is_power
                        # Avoid complex numbers. Not really sure what to do for this case.
                        # I guess for backprop, this part isn't really needed anyway, its just here for completeness
                        # (a param isn't really ever a power)
                        if base.data == 0:
                            child.grad = 0
                        elif base.data < 0:
                            child.grad += (
                                -math.log(abs(base.data)) * base.data**power.data
                            ) * self.grad
                        else:
                            child.grad += (
                                math.log(base.data) * base.data**power.data
                            ) * self.grad

                case Operation.RELU:
                    assert len(self._children) == 1
                    child.grad += self.grad if self.data > 0 else 0

                case Operation.TANH:
                    assert len(self._children) == 1
                    child.grad += (1 - self.data**2) * self.grad

                case Operation.EXP:
                    assert len(self._children) == 1
                    child.grad += self.data * self.grad

    def get_children_ordered(self):
        ordered_nodes: List["Value"] = [self]
        children_queue: List["Value"] = [self]
        while len(children_queue) != 0:
            current = children_queue.pop(0)
            ordered_nodes = [*ordered_nodes, *current._children]
            children_queue = [*children_queue, *current._children]

        return ordered_nodes

    def backward(self):
        self.grad = 1.0
        ordered_nodes = self.get_children_ordered()
        for value in ordered_nodes:
            value._backward()

    def relu(self):
        out = self.data
        if out <= 0:
            out = 0
        return Value(
            out, _children=(self,), label=f"relu({self.label})", _op=Operation.RELU
        )

    def tanh(self):
        return Value(
            math.tanh(self.data),
            _children=(self,),
            label=f"tanh({self.label})",
            _op=Operation.TANH,
        )

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: accepted_types):
        if not isinstance(other, Value):
            other = Value(other)
        value = Value(
            self.data + other.data, _children=(self, other), _op=Operation.ADD
        )
        return value

    def __mul__(self, other: accepted_types):
        if not isinstance(other, Value):
            other = Value(other)
        value = Value(
            self.data * other.data, _children=(self, other), _op=Operation.MUL
        )
        return value

    def __pow__(self, other: accepted_types):
        if not isinstance(other, Value):
            other = Value(other)
        value = Value(self.data**other.data, _children=(self, other), _op=Operation.POW)
        return value

    def __truediv__(self, other: accepted_types):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other: accepted_types):
        return self + (-other)

    def __radd__(self, other: accepted_types):
        return self + other

    def __rsub__(self, other: accepted_types):
        return (-self) + other

    def __rmul__(self, other: accepted_types):
        return self * other

    def __rpow__(self, other: accepted_types):
        if not isinstance(other, Value):
            other = Value(other)
        return other**self

    def __rtruediv__(self, other: accepted_types):
        return self**-1 * other
