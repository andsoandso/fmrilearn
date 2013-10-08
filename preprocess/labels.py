"""Functions for preprocessing (i.e. filtering, combining) and aggregating
labels."""
import numpy as np

from copy import deepcopy

def construct_targets(**kwargs):
    """Construct a dictionary of training targets.
    
    Parameters
    ----------
    A variable number of keyword arguments.  
        The key is the name the value is a list/array of targets.

    Note
    ----
    Lengths of lists/arrays must be equal
    """
    
    # Easier, though less efficient, to build targets
    # then check the length.
    targets = {}
    for key, val in kwargs.items():
        targets[key] = val

    lenv = len(targets.values()[0])
    for v in targets.values():
        if len(v) != lenv:
            raise ValueError("Target value length mismatch")

    return targets


def merge_labels(labels, labelmap):
    """Merge labels. 
    
    Parameters
    ----------
    labels - a 1d array of class labels
    labelmap - a dict whose keys are the old labels and whose values are the 
        new labels.
    
    Returns
    -------
    The merged labels (1d array).  
    
    Note
    ----
    Labels that do not match a key in labelmap are converted to None or
        'np.nan'
    """
    
    # Create the new labels
    newlabels = []
    for label in labels:
        try:
            new = labelmap[label]
        except KeyError:
            new = None
        newlabels.append(new)

    return np.array(newlabels)


def join_labels(*args):
    """Join the N given labels.
    
    Parameters
    ----------
    (labels1, labels2 ...) Combined N given lists into a single list, where
    each element is joined by an underscore.
    
    Returns
    -------
    The joined labels (1d array)
    """
    
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


def locate_short_trials(trial_index, window):
    """Return the codes of trials less than window"""
    
    # Is the i'th trial less than the window,
    # if yes add it to the list.
    short_trials = []
    for i in np.unique(trial_index):
        if np.sum(i == trial_index) < window:
            short_trials.append(i)
    
    return short_trials
    

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
