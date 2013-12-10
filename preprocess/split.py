import numpy as np
from fmrilearn.preprocess.labels import unique_sorted_with_nan

def by_labels(X, y):
    """Split up X into a list if Xs, one for each unique entry in y.

    Parameters
    ----------
    X : a 2d array-like (n_features x n_sample)
        The feature matrix
    y : a 1d array-like
        The labels for X

    Return
    ------
    Xs : seq-like, containing 2D arrays (n_feature x n_sample)
        The split up Xs
    ys : seq-like, containing 1D arrays
        The y matching each X in Xs
    """

    # ----`
    # Find all samples
    y_masks = []
    unique_y = sorted(np.unique(y))
    unique_y = unique_sorted_with_nan(unique_y)

    for y_i in unique_y:
        y_masks.append(np.str(y_i) == y)
    
    # And split each feature into seperate Xs
    Xs = []
    ys = []
    for mask in y_masks:
        Xs.append(X[mask,])
        ys.append(y[mask])

    return Xs, ys
