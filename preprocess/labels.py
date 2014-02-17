"""Functions for preprocessing (i.e. filtering, combining) and aggregating
labels."""
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from json import load


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


def targets_to_csv(targets, name, header=True):
    """Write targets to a .csv file.

    Parameters
    ---------
    targets : dict
        A dictionary of training of other metadata targets
    name : str
        The csv name (must add '.csv' manually)
    header : bool
        Should we write a header matching targets' keys?
    """

    l = len(targets.values()[0])
    print(l)
    head = targets.keys()
    head.sort()

    with open(name, 'w') as f:
        csvf = csv.writer(f, delimiter=",")
        if header:
            csvf.writerow(head)
            
        for i in range(l): 
            csvf.writerow([str(targets[k][i]) for k in head])         
        

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


def tr_pad_targets(targets, tr, nrow, pad=np.nan):
    """Pad all targets to match the number of rows starting with 0
    
    Parameters
    ----------
    targets : dict
        The named (by key) classification targets.
    tr : str
        The name of the target in targets that has the TR metadata
    nrow : int
        The number of of rows in the data (presumable X) were padding
        to match
    pad : str, optional
        What to use as padding. The 'tr' entry in the new targets
        is not padded.  It is ints `[0 ... nrow-1]` instead.
    """

    # If the len match there is nothing to do
    if targets[tr].shape[0] == nrow:
        return targets

    # Construct new targets
    # that are filled with padding, 
    # but matching type
    # of the orginal targets
    #
    # Then use tr from targets
    # to insert values from
    # targets to padded
    padded = {}
    padded[tr] = np.arange(nrow)
    for k, v in targets.items():
        if k != tr:
            padded[k] = np.empty(nrow, dtype=v.dtype)
            padded[k].fill(pad)
            for i, tr_i in enumerate(targets[tr]):
                padded[k][tr_i] = v[i]
    return padded


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
    
    filtered = {}
    for key, val in targets.items():
        filtered[key] = val[index]
    return filtered


def replace_targets(index, targets, value=np.nan, exclude=("TR",)):
    """Use the index to replace targets with value"""

    replaced = deepcopy(targets)
    for key in targets.keys():
        if key in exclude:
            continue
        else:
            replaced[key][index] = value

    return replaced


def reprocess_targets(name, targets, value=None):
    """ Use a config file to filter targets.

    Parameters
    ----------
    name - str, a file path
        The name of valid json file (see Info)
    targets - dict-like
        A dictionary of labels/targets for X. Keys 
        are names and values are sklearn compatible
        lebels
    value : str, object
        Value to replace not kept entries with

    Return
    ------
    The filtered X, targets

    Info
    ----
    The named json file has the must can only have 3 
    top level nodes ["keep", "merge", "join"], one of
    which must be present.

    Below each top-level key is a label name which must be 
    present in targets.

    From there it depends on which top-level branch you are in
        TODO
    """

    # load the json at name,
    filterconf = load(open(name, "r"))

    # Validate top level nodes
    validnodes = ["keep", "merge", "join"]
    for k in filterconf.keys():
        if k not in validnodes:
            raise ValueError("Unknown filter command {0}".format(k))

    # test for keep and do that
    if "keep" in filterconf:
        for k, keepers in filterconf["keep"].items():
            labels = targets[k] 
            mask = construct_filter(labels, keepers, True)

            # Invert the mask
            # and use the inv matrix to replace
            invmask = np.delete(np.arange(labels.shape[0]), mask) 
            targets = replace_targets(invmask, targets, value) # DEBUG

    # Test for merge and do that
    if "merge" in filterconf:
        for k, mmap in filterconf["merge"].items():
            labels = targets[k]
            targets[k] = merge_labels(labels, mmap)   

    # Test for join and do that
    if "join" in filterconf:
        raise NotImplementedError("join not yet implemented.  Sorry.")

    return targets                                    


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
