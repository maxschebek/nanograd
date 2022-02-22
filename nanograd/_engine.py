from typing import Union, get_args

Data = Union[int, float]
MaybeValue = Union["Value", Data]


class Value:
    def __init__(self, data: Data, children=(), operation=""):
        self.data = data
        self.children = children
        self.operation = operation
        self.grad = 0
        self.compute_children_grad = lambda: None

    def __repr__(self):
        return f"V({self.data})"

    def __add__(self, other: MaybeValue):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def compute_children_gradient():
            self.grad += out.grad
            other.grad += out.grad

        out.compute_children_grad = compute_children_gradient
        return out

    __radd__ = __add__

    def __mul__(self, other: MaybeValue):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def compute_children_gradient():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.compute_children_grad = compute_children_gradient
        return out

    def __pow__(self, other: Data):
        if not isinstance(other, get_args(Data)):
            raise NotImplementedError("Exponent must be 'int' or 'float' type.")

        out = Value(self.data ** other, (self,), "**")

        def compute_children_gradient():
            self.grad += out.grad * other * self.data ** (other - 1)

        out.compute_children_grad = compute_children_gradient
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other: MaybeValue):
        return self + (-other)

    def __rsub__(self, other: MaybeValue):
        return (-self) + other

    def __truediv__(self, other: MaybeValue):
        return self * other ** -1

    def __rtruediv__(self, other: MaybeValue):
        return self ** -1 * other

    def relu(self):
        out = Value(max(0, self.data), (self,), "ReLU")

        def compute_children_gradient():
            self.grad += (out.data > 0) * out.grad

        out.compute_children_grad = compute_children_gradient
        return out

    @staticmethod
    def _flatten_tree(value: "Value", visted=set()):
        for child in value.children:
            if child not in visted:
                visted.add(child)
                yield from Value._flatten_tree(child)
        # add itself after adding children and then reverse ordered values
        # so that in the case of duplicate dependencies the gradients are computed
        # in the right order
        yield value

    def backward(self):
        self.grad = 1
        for value in reversed(list(Value._flatten_tree(self))):
            value.compute_children_grad()
