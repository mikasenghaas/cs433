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

    # gradient step
    w -= lr * g

    pbar.set_description(
            f"[{str(epoch+1).zfill(len(str(epochs)))}/{epochs} - Loss: {loss:.4f}"
    )
 
print("\nTraining End")
print(f"Loss: {mse(yt, lm(xt, w))} (w1 = {w[0]}, w2 = {w[1]})")
