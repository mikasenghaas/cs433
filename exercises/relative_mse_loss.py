import numpy as np
from matplotlib import pyplot as plt

def mse_loss(y_hat, y):
    """
    Computes the loss between two arrays of equal size.
    """
    return np.mean(np.square(y_hat - y))

def relative_mse_loss(y_hat, y):
    """
    Computes the loss between two arrays of equal size.
    """
    epsilon = 1e-5
    return np.mean(np.square(y_hat - y) / (np.square(y)+epsilon))

def plot_relative_mse_loss():
    """
    Plots the relative MSE loss for a range of values in 3D space
    for a range of prediction + target pairs
    """
    r = 10
    xn, yn = 10, 10
    ys = np.linspace(-r, r, xn)
    y_hats = np.linspace(-r, r, yn)
    ys, y_hats = np.meshgrid(ys, y_hats)
    losses = [relative_mse_loss(y_hat,y) for y_hat, y in zip(ys.ravel(), y_hats.ravel())]
    losses = np.array(losses).reshape(xn, yn)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ys, y_hats, losses)
    ax.set_xlabel('y')
    ax.set_ylabel('y_hat')
    ax.set_zlabel('loss')
    plt.show()

def main():
    plot_relative_mse_loss()

if __name__ == "__main__":
    main()
