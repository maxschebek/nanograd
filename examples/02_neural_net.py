#%%
import random

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12234)
random.seed(2237)


# make up dataset
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=100, noise=0.1)

y = y * 2 - 1  # make y be -1 or 1
# visualize in 2D
# plt.figure(figsize=(5, 5))
# plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
# plt.savefig("neural_network_")

from micrograd.engine import Value

# initialize model
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])
# print(model)
print("number of parameters", len(model.parameters()))

# define loss function
def loss(batch_size=None):
    if batch_size is None:
        xb, yb = x, y
    else:
        ri = np.random.permutation(x.shape[0])[:batch_size]
        xb, yb = x[ri], y[ri]

    inputs = [[Value(value) for value in row] for row in xb]
    scores = [model(input_) for input_ in inputs]

    # svm max-margin loss
    losses = [(1 - yi * score_i).relu() for yi, score_i in zip(yb, scores)]
    data_loss = sum(losses) / len(losses)

    # l2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum(p * p for p in model.parameters())
    total_loss = data_loss + reg_loss

    # get accuracy
    accuracy = sum(
        (yi > 0) == (score_i.data > 0) for yi, score_i in zip(yb, scores)
    ) / len(yb)
    return total_loss, accuracy


print(loss())

#%% train
for k in range(50):
    total_loss, accurary = loss()

    model.zero_grad()
    total_loss.backward()

    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(f"step {k} loss {total_loss.data}, accuracy {accurary*100}%")

#%%
h = 0.25
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

from pathlib import Path

plt.savefig(Path(__file__).parent / "classification_moon.svg")
