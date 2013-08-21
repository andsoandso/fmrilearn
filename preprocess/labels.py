"""Functions for preprocessing (i.e. filtering, combining) and aggregating
labels."""
import numpy as np

from copy import deepcopy

def construct_targets(**kwargs):
    """Construct a dictionary of training targets.
    
    Parameters:
    ----------
    input a variable number of keyword arguments.  The key is the name the
    value is the list/array of targets.
    """
    
    targets = {}
    for key, val in kwargs.items():
        targets[key] = val
    
    return targets


def merge_labels(*args):
    """Combined the given lists into a single list,
    where each element is joined by an underscore."""
    
    args = list(args)
    merged = args.pop(0)
    for labels in args:
        if len(merged) != len(labels):
            raise ValueError("Lengths did not match")

        tmp = ["{0}_{1}".format(m, x) for m, x in zip(merged, labels)] 
        merged = deepcopy(tmp)

    return np.array(merged)


def filter_targets(index, targets):
    """Use the index to filter targets and return the result."""
    
    ftargets = {}
    for key, val in targets.items():
        ftargets[key] = val[index]
    
    return ftargets


def create_y(labels):
    """Generate y from labels."""

    # Create y, a vector of ints
    # matching sorted entries in labels
    uniq = sorted(set(labels))
    labels = np.array(labels)
    y = np.zeros_like(labels, dtype=int)
    for ii, lab in enumerate(uniq):
        mask = labels == lab 
        y[mask] = ii

    return y


def construct_filter(labels, keepers, indices=True): 
    labels = np.array(labels)
    keepers = list(keepers)
    
    # Init then iterate
    mask = labels == keepers.pop()
    for keep in keepers:
        mask = mask | (labels == np.array(keep))
    
    if indices:
        mask = np.arange(labels.shape[0])[mask]
    
    return mask 
