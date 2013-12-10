"""Functions for preprocessing (i.e. filtering, combining) and aggregating
labels."""
import numpy as np
import pandas as pd
from copy import deepcopy


def csv_to_targets(csvf):
    """Convert a csv file to a targets dictionary.

    Note
    ----
    Assumes each colummn is a target (i.e. a list of labels), 
    and that the colname name should be used as the key.
    """

    # Convert then array-ify
    targets = pd.read_csv(csvf).to_dict(outtype='list')
    for k, v in targets.items():
        targets[k] = np.array(v)
    
    return targets


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
    uniq = sorted(np.unique(labels))
    uniq = unique_sorted_with_nan(uniq)

    print("\ty map: "+ ",".join(
            ["{0} -> {1}".format(lab, ii) for ii, lab in enumerate(uniq)]))    
    
    labels = np.array(labels)
    y = np.zeros_like(labels, dtype=int)
    for ii, lab in enumerate(uniq):
        mask = labels == np.str(lab) 
        y[mask] = ii

    return y


def unique_sorted_with_nan(unique_list):
    """Given a unique sorted list (not a set()) resorted the list
    treating 'nan' as null
    
    Parameters
    ---------
    unique_list - sorted(list)
        A list whoe elements are unique and sorted
        by sorted() or an equivilant.

    Note
    ---
    A unique_list could be create by 
        >>> sorted(list(np.unique(list)))
    or 
        >>> sorted(list(set(list)))
    """

    unique_list = list(unique_list)
    if sorted(unique_list) != unique_list:
        raise ValueError("unique_list wasn't sorted")

    if len(unique_list) != len(list(np.unique(unique_list))):
        raise ValueErro("unique_list wasn't unique")

    ## np.nan is sometime converted to 'nan'.
    ## but if should be first in uniq 
    nanindex = None
    try:
        nanindex = unique_list.index('nan')
    except ValueError:
        pass
    if nanindex is not None:
        unique_list.insert(0, unique_list.pop(nanindex)) 
    
    return unique_list


def locate_short_trials(trial_index, window):
    """Return the codes of trials less than window"""
    
    # Is the i'th trial less than the window,
    # if yes add it to the list.
    short_trials = []
    for i in np.unique(trial_index):
        if np.isnan(i): continue
        if np.sum(i == trial_index) < window:
            short_trials.append(i)
    
    return short_trials
    

def construct_filter(labels, keepers, indices=True): 
    labels = np.array(labels)
    keepers = list(keepers)
    
    # Init then iterate
    mask = labels == np.str(keepers.pop())
    for keep in keepers:
        mask = mask | (labels == np.array(keep))
    
    if indices:
        mask = np.arange(labels.shape[0])[mask]
    
    return mask 
