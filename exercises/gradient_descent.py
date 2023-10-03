import os
import numpy as np
from tqdm import trange

# true function
f = lambda x: x @ np.array([-10, 3])

# linear model
lm = lambda x, w: x @ w

# loss
mse = lambda y, y_hat: np.mean(np.square(y - y_hat))

# gradient of linear mse
glmse = lambda x, w, y: - 1 / len(y) * x.T @ (y - x @ w)

n = 1000
xt = np.hstack((np.random.uniform(-10, 10, size=(n, 1)),np.random.uniform(-10, 10, size=(n, 1))))
yt = f(xt) + np.random.randn(n)

# training hyperparams
epochs = 100
lr = 1e-4

losses = {"gd": [], "sgd": [], "sgdm": []}
for m in range(5):
    # initialise model linear model weights
    wo = np.random.randn(2)
    w = wo.copy()


    # training with gd
    for epoch in range(epochs):
        preds = lm(xt, w)
        loss = mse(yt, preds)
        g = glmse(xt, w, yt)

        # gradient step
        w -= lr * g

    loss_gd = mse(yt, lm(xt, w))
    losses["gd"].append(loss_gd)

    # training with sgd
    w = wo.copy()

    for epoch in range(epochs):
        # sample a random data point
        idx = np.random.randint(0, len(yt))
        xts, yts = xt[idx], yt[idx]

        preds = lm(xts, w)
        loss = mse(yts, preds)
        g = glmse(xt, w, yt)

        # gradient step
        w -= lr * g

    loss_sgd = mse(yt, lm(xt, w))
    losses["sgd"].append(loss_sgd)

    # training with sgd w/ momentum
    w = wo.copy()
    m = np.zeros_like(w)
    beta = 0.9


    for epoch in range(epochs):
        # sample a random data point
        idx = np.random.randint(0, len(yt))
        xts, yts = xt[idx], yt[idx]

        preds = lm(xts, w)
        loss = mse(yts, preds)
        g = glmse(xt, w, yt)

        # gradient step with momentum 
        m = beta * m + (1 - beta) * g
        w -= lr * m

    loss_sgdm = mse(yt, lm(xt, w))
    losses["sgdm"].append(loss_sgdm)

print("gd: ", np.mean(losses["gd"]))
print("sgd: ", np.mean(losses["sgd"]))
print("sgdm: ", np.mean(losses["sgdm"]))
