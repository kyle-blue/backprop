from typing import Set


class Value:
    def __init__(self, data: float, label="", _children=(), _op=""):
        self.data = data
        self.label = label
        self._children: Set[Value] = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other: "Value"):
        value = Value(self.data + other.data, _children=(self, other), _op="+")
        return value

    def __mul__(self, other: "Value"):
        value = Value(self.data * other.data, _children=(self, other), _op="*")
        return value
