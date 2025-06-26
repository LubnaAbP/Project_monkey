import numpy as np
import scipy
import scipy.signal
from math import ceil
import itertools
import os
import pickle
from tqdm import tqdm




# File paths for saving/loading smoothed and centered data

def smoothing(X: np.array, bin_size=0.025, K=2, width=1.5):
    """
    Applies exponential smoothing to spike data (one session).

    Parameters:
    X : np.array
        Spike data (trials x neurons x time bins)
    bin_size : float
        Width of each time bin (in seconds)
    K : float
        Shape parameter for exponential decay
    width : float
        Controls smoothing extent (in seconds)

    Returns:
    X_smoothed : np.array
        Smoothed spike data
    """
    bin_w = int(ceil(width / bin_size))
    win = scipy.signal.windows.exponential(2 * bin_w + 1, tau=bin_w / (2 * K))
    win[:bin_w] = 0
    win /= win.sum() * bin_size  # Normalize for area under the curve

    new_data = np.zeros_like(X)
    convol_fun = lambda x: np.convolve(x, win, mode='same')

    for c, n in itertools.product(range(X.shape[0]), range(X.shape[1])):
        new_data[c, n, :] = convol_fun(X[c, n, :])

    return new_data




