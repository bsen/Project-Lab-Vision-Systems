import numpy as np
import matplotlib.pyplot as plt

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def plot_losses(train_loss, loss_iters, val_loss, val_err):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("SmoothL1 Loss")
    ax[0].set_title("Training Progress")

    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs[1:], train_loss[1:], c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs[1:], val_loss[1:], c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("SmoothL1 Loss")
    ax[1].set_title("Loss Curves")

    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs[1:], val_err[1:], c="red", label="Validation 3PE", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Error (%)")
    ax[2].set_title(f"Valdiation 3PE (max={round(np.min(val_err),2)}% @ epoch {np.argmin(val_err)+1})")

    plt.show()
