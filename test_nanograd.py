from nanograd import Value


def test_arithmetic():
    a = Value(1)
    b = Value(2)
    assert 1 == a.data
    assert 2 == b.data
    assert 3 == (a + b).data
    assert -1 == (-a).data
    assert -1 == (a - b).data
    assert 2 == (a * b).data

    # test left-hand arithemtic with numbers
    assert 3 == (a + 2).data
    assert -1 == (a - 2).data
    assert 2 == (a * 2).data

    # test right-hand arithmetic with numbers
    assert 3 == (1 + b).data
    assert -1 == (1 - b).data
    assert 2 == (1 * b).data


def test_flatten_tree_without_duplicate_dependencies():
    a = Value(1)
    b = Value(2)
    c = Value(3)
    d = Value(4)
    e = a + b
    f = c + d
    g = e + f
    assert [a, b, e, c, d, f, g] == list(Value._flatten_tree(g))


def test_flatten_tree_with_duplicate_dependencies():
    a = Value(1)
    b = Value(2)
    c = Value(3)
    d = a + b
    e = d + c
    f = d + e
    assert [a, b, d, c, e, f] == list(Value._flatten_tree(f))


def test_grad_add():
    a = Value(1)
    b = Value(2)
    c = a + b
    c.backward()
    assert 1 == c.grad
    assert 1 == a.grad
    assert 1 == b.grad


def test_grad_sub():
    a = Value(2)
    b = Value(3)
    c = a - b
    c.backward()
    assert 1 == c.grad
    assert 1 == a.grad
    assert -1 == b.grad


def test_grad_mul():
    a = Value(2)
    b = Value(3)
    c = a * b
    c.backward()
    assert 1 == c.grad
    assert 3 == a.grad
    assert 2 == b.grad


def test_engine_from_micrograd():
    # https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    assert 46.0 == x.grad
    assert -20.0 == y.data

    # test_more_op
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    g.backward()

    tol = 1e-6
    assert abs(24.70408163265306 - g.data) < tol
    assert abs(138.83381924198252 - a.grad) < tol
    assert abs(645.5772594752186 - b.grad) < tol


def test_neuron():
    from nanograd import Neuron

    neuron = Neuron(5, relu=True)
    assert "Neuron(5,ReLU)" == repr(neuron)
    neuron = Neuron(5, relu=False)
    assert "Neuron(5,Linear)" == repr(neuron)


def test_layer():
    from nanograd import Layer

    layer = Layer(3, 4, relu=True)
    assert "Layer(3,4)" == repr(layer)


def test_mlp():
    from nanograd import MLP

    mlp = MLP([3, 4, 3])
    assert "MLP(3,4,3)" == repr(mlp)
