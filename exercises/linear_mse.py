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

def glmse2(x, w, y):
    dLdw_i = [-1 / len(y) * x[:, i].T @ (y - x @ w) for i in range(x.shape[1])]
    return np.array(dLdw_i)

n = 1000
xt = np.hstack((np.random.uniform(-10, 10, size=(n, 1)),np.random.uniform(-10, 10, size=(n, 1))))
yt = f(xt) + np.random.randn(n)

# initialise model linear model weights
w = np.random.randn(2)

# training loop
epochs = 10000
lr = 1e-4
print(f"Starting Training")
print(f"Loss: {mse(yt, lm(xt, w))} (w1 = {w[0]}, w2 = {w[1]})")

pbar = trange(epochs)
for epoch in pbar:
    preds = lm(xt, w)
    loss = mse(yt, preds)
    g = glmse(xt, w, yt)
    g2 = glmse2(xt, w, yt)
    print(g, g2)
    assert np.allclose(g, g2), "Gradient computations are not equal"

    # gradient step
    w -= lr * g

    pbar.set_description(
            f"[{str(epoch+1).zfill(len(str(epochs)))}/{epochs} - Loss: {loss:.4f}"
    )
 
print("\nTraining End")
print(f"Loss: {mse(yt, lm(xt, w))} (w1 = {w[0]}, w2 = {w[1]})")
