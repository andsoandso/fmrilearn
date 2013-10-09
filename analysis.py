"""Functions to analyze features."""
import numpy as np
import nitime.timeseries as ts
import nitime.analysis as nta

from copy import deepcopy
from scipy.stats  import spearmanr, pearsonr

from sklearn.preprocessing import scale

from fmrilearn.info import print_X_info
from fmrilearn.info import print_label_counts
from fmrilearn.preprocess.labels import locate_short_trials
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.reshape import by_trial
from fmrilearn.preprocess.split import by_labels


def _create_cond_y(conds, trial_length):
    # Finally create the cond labels
    y = []
    for cond in conds:
        y.extend([cond, ] * trial_length)

    return np.array(y)


def correlateX(X, y, corr="spearman"):
    """Correlate each feature in X, with y (some set of dummmy 
        coded labels).
     
    Parameters
    ----------
    X - a 2d col oreinted array of features
    y - a 1d array of labels
    corr - name of correlation function:
        'pearson' or 'spearman'
    
    Returns
    -------
    corrs - a 1d array of correlations
    ps - a 1d array of p-values
    
    Note
    ----
    Correlation's are calculated using either pearson's r (which 
    assumes Gaussian errors) of spearman's rho (a rank-based 
    non-parametric method.)
    """
    
    X = np.array(X)
    y = np.array(y)
        ## Force... just in case
    
    checkX(X)

    if corr == "pearson":
        corrf = pearsonr
    elif corr == "spearman":
        corrf = spearmanr
    else:
        raise ValueError("stat was not valid.")
    
    corrs = []
    ps = []
    for jj in range(X.shape[1]):
        r, p = corrf(X[:,jj], y)
        corrs.append(r)
        ps.append(p)
        
    return np.array(corrs), np.array(ps)


def eva(X, y, trial_index, window, norm=True):
    evas = []
    eva_names = []
    unique_y = sorted(np.unique(y))
    for j in range(X.shape[1]):
        Xtrials = []
        
        xj = X[:,j][:,np.newaxis]  ## Need 2D

        # Each feature into trials
        Xtrial, feature_names = by_trial(xj, trial_index, window, y)
        if norm:
            scale(Xtrial.astype(np.float), 
                    axis=0, with_mean=False, copy=False)

        # and again by unique_y/feature_names
        Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

        # put all that togthether
        Xtrials.append(Xtrial)
        Xtrials.extend([Xl.transpose() for Xl in Xlabels])

        # and average the trials.
        evas.extend([Xt.mean(axis=1) for Xt in Xtrials])

        # Name names.
        eva_names.extend(["all", ] + unique_y)

    # Reshape : (window, len(unique_y)*n_features)
    Xeva = np.vstack(evas).transpose()
    eva_names = np.asarray(eva_names)
    
    assert checkX(Xeva)
    assert Xeva.shape[0] == window, ("After EVA rows not equal to window")
    assert Xeva.shape[1] == ((len(unique_y) + 1) * X.shape[1]), ("After" 
        "EVA wrong number of features")
    assert eva_names.shape[0] == Xeva.shape[1], ("eva_names and Xeva" 
        "don't match")

    return Xeva, eva_names

