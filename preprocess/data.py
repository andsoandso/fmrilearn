"""Functions for preprocessing fmrilearn data."""

import numpy as np

from scipy.sparse import csc_matrix

import scipy.signal as signal
import nitime as nt


def find_good_features(X, sparse=True):
    """Return an index of features (cols) with non-zero and 
    non-constant values."""

    # If the col contains *an* non-zero entry keep it.
    if sparse:
        keepcol = np.unique(X.tocoo().col)
            ## The COO format has a list of occupied
            ## cols built in.  Keep unique only.
    else:
        sds = X.std(axis=0)
        keepcol = np.arange(X.shape[0])[sds > 0.0001]
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

    return Xfea, np.array(newlabels)