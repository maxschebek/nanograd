from random import uniform
from typing import List
from itertools import chain

from ._engine import Value, MaybeValue


class Neuron:
    def __init__(self, nin, relu):
        self.weights = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(0)
        self.relu = relu
        self.parameters = self.weights + [self.bias]

    def __call__(self, x: List[MaybeValue]) -> MaybeValue:
        result = sum(wi * xi for (wi, xi) in zip(self.weights, x)) + self.bias
        return result.relu() if self.relu else result

    def __repr__(self) -> str:
        return f"Neuron({len(self.weights)},{'ReLU' if self.relu else 'Linear'})"

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0


class Layer:
    def __init__(self, n_in: int, n_out: int, relu: bool) -> None:
        self.neurons = [Neuron(n_in, relu) for _ in range(n_out)]
        self.parameters = list(
            chain.from_iterable(neuron.parameters for neuron in self.neurons)
        )

    def __call__(self, x: List[MaybeValue]) -> List[MaybeValue]:
        return [neuron(x) for neuron in self.neurons]

    def __repr__(self) -> str:
        n_in, n_out = len(self.neurons[0].weights), len(self.neurons)
        return f"Layer({n_in},{n_out})"

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0


class MLP:
    def __init__(self, dimensions: List[int]):
        self.dimensions = dimensions
        self.layers = [
            Layer(dim_in, dim_out, relu=i != len(dimensions) - 2)
            for i, (dim_in, dim_out) in enumerate(zip(dimensions, dimensions[1:]))
        ]
        self.parameters = list(
            chain.from_iterable(layer.parameters for layer in self.layers)
        )

    def __call__(self, x: List[MaybeValue]) -> List[MaybeValue]:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        return f"MLP({','.join(str(dimension) for dimension in self.dimensions)})"

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0
