from micrograd.engine import Value

x = Value(-4.0)
y = Value(2.0)

f = x * y ** 2
f.backward()
g = 2 * f
print(x.grad)
print(y.grad)
print(g.data)

h = 2 * g
h.backward()
print(g.grad)
