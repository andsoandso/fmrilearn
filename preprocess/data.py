"""Functions for preprocessing fmrilearn data."""

from json import load
import numpy as np

import nitime as nt
import scipy.signal as signal
from scipy.sparse import csc_matrix

from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets
from fmrilearn.preprocess.labels import merge_labels


def filterX(filtname, X, targets):
    """ Use a config file to filter both X and targets.

    Parameters
    ----------
    filtname - str, a file path
        The name of valid json file (see Info)
    X - 2D array-like (n_samples x n_features)
        The data to filter
    targets - dict-like
        A dictionary of labels/targets for X. Keys 
        are names and values are sklearn compatible
        lebels

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
    filterconf = load(open(filtname, "r"))

    # Validate top level nodes
    validnodes = ["keep", "merge", "join"]
    for k in filterconf.keys():
        if k not in validnodes:
            raise ValueError("Unknown filter command {0}".format(k))

    # Validate that X and targets match
    for k, v in targets.items():
        if v.shape[0] != X.shape[0]:
            raise ValueError("Before: X/target shape mismatch for '{0}'".format(k))

    # test for keep and do that
    if "keep" in filterconf:
        for k, keepers in filterconf["keep"].items():
            labels = targets[k] 
            mask = construct_filter(labels, keepers, True)
            targets = filter_targets(mask, targets)
            X = X[mask,:]

    # Test for merge and do that
    if "merge" in filterconf:
        for k, mmap in filterconf["merge"].items():
            labels = targets[k]
            targets[k] = merge_labels(labels, mmap)   

    # Test for join and do that
    if "join" in filterconf:
        raise NotImplementedError("join not yet implemented.  Sorry.")

    # revalidate that X and targets match
    for k, v in targets.items():
        if v.shape[0] != X.shape[0]:
            raise ValueError("After: X/targets shape mismatch for '{0}'".format(k))
    assert checkX(X)
    
    return X, targets


def checkX(X):
    """Is X OK to use?

    Return
    ------
    status: bool, Exception
        True if OK, error otherwise.
    """

    status = False
    if not hasattr(X, "shape"):
        raise TypeError("X must be array-like")
    elif X.size == 0:
        raise ValueError("X is empty")
    elif X.ndim != 2:
        raise ValueError("X must be 2d.")
    elif np.isnan(X).any():
        raise ValueError("X contains NaNs")
    elif np.isinf(X).any():
        raise ValueError("X contains Inf")
    else:
        status = True

    return status


def find_good_features(X, sparse=True, tol=0.001):
    """Return an index of features (cols) with non-zero and 
    non-constant values."""

    # If the col contains *an* non-zero entry keep it.
    if sparse:
        keepcol = np.unique(X.tocoo().col)
            ## The COO format has a list of occupied
            ## cols built in.  Keep unique only.
    else:
        sds = X.std(axis=0)
        mask = sds > tol 
        keepcol = np.arange(X.shape[1])[mask]
            ## For non-sparse remove cols with extremely
            ## low standard deviations.  A different
            ## approach than for sparse, but for fMRI
            ## data it should have a similar effect....
            ## BUT HAS NOT BEEN TESTED.

    return keepcol


def remove_invariant_features(X, sparse=True):
    """Remove invariant features (i.e columns) in X. 
    
    Note: If sparse is True, the sparsity is destroyed
    by this function, but that is fine.  The point of this
    function is to eliminate the zeros.
    """
    
    keepcol = find_good_features(X, sparse)
    X = X[:,keepcol]
    
    assert checkX(X)

    if sparse:
         X = csc_matrix(X)

    return X


def smooth(X, tr=1.5, ub=0.10, lb=0.001):
    """Smooth columns in X.
    
    Parameters:
    -----------
    X - a 2d array with features in cols
    tr - the repetition time or sampling (in seconds)
    ub - upper bound of the band pass (Hz/2*tr)
    lb - lower bound of the band pass (Hz/2*tr)

    Note:
    ----
    Smoothing is a linear detrend followed by a bandpass filter from
    0.0625-0.15 Hz
    """
    
    # Linear detrend
    Xf = signal.detrend(X, axis=0, type='linear', bp=0)
    
    # Band pass
    ts = nt.TimeSeries(Xf.transpose(), sampling_interval=tr)      
    Xf = nt.analysis.FilterAnalyzer(ts, ub=ub, lb=lb).fir.data
        ## ub and lb selected after some experimentation
        ## with the simiulated accumulator data 
        ## ub=0.10, lb=0.001).fir.data
    
    Xf = Xf.transpose()
        ## TimeSeries assumes last axis is time, and we need
        ## the first axis to be time.
    
    return Xf


def shiftby(X, targets, by):
    """Accounts for HRF lag. Shift X (a single array) and targets 
    (a list of ys and labs) by <by>. """
    
    by = int(by)
    
    # Do nothing when by is 0,
    if by == 0:
        return X, targets
    
    # otherwise shift rows down by
    X = X[by:,:]  

    # and targets drop by for constant l
    for key, tar in targets.items():
        targets[key] = tar[0:(tar.shape[0] - by)] 

    assert checkX(X)

    return X, targets


def create_X_stats(X, trial_index, labels):
    """Generate trial-level statistics for every feature in X."""
    
    trials = np.unique(trial_index)

    # ----
    # Init
    Xmax = np.zeros((trials.shape[0], X.shape[1]))
    Xmin = np.zeros_like(Xmax)
    Xmean = np.zeros_like(Xmax)
    Xvar = np.zeros_like(Xmax)

    # ----
    # Create the stats
    newlabels = []
    for ii, trial in enumerate(trials):
        # Locate this trials data
        mask = trial == trial_index
        x_trial = X[mask,:]

        # Get time to peak/min
        Xmax[ii,:] = np.argmax(x_trial, axis=0)
        Xmin[ii,:] = np.argmin(x_trial, axis=0)
        
        # And their diff
        Xdiff = Xmax - Xmin
        
        # Finally get trial means and variances
        Xmean[ii,:] = x_trial.mean()
        Xvar[ii,:] = x_trial.var()

        # Only need one label for each trial
        # now, the first is as good as any
        newlabels.append(labels[mask][0])
    
    Xfea = np.hstack([Xmax, Xmin, Xdiff, Xmean, Xvar])

    assert checkX(Xfea)
    
    return Xfea, np.array(newlabels)
