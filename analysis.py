"""Functions to analyze features."""
import numpy as np
import nitime.timeseries as ts
import nitime.analysis as nta

from copy import deepcopy
from scipy.stats  import spearmanr, pearsonr

from sklearn.preprocessing import MinMaxScaler

from fmrilearn.info import print_X_info
from fmrilearn.info import print_label_counts
from fmrilearn.preprocess.labels import locate_short_trials
from fmrilearn.preprocess.labels import create_y
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


def fir(X, y, trial_index, window, tr):
    """ Average trials for each feature in X, using Burock's 
    (2000) method.
    
    Parameters
    ----------
    X : 2D array-like (n_sample, n_feature)
        The data to decompose
    y : 1D array, None by default
         Sample labels for the data. In y, np.nan and 'nan' values 
         are treated as baseline labels.
    trial_index : 1D array (n_sample, )
        Each unique entry should match a trial.
    window : int 
        Trial length

    Return
    ------
    Xfir : a 2D arrays (n_feature*unique_y, window)
        The average trials
    feature_names : 1D array
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.astype(np.float))

    ynames = sorted(np.unique(y))
    y = create_y(y)
    ty = ts.TimeSeries(y, sampling_interval=tr)
            ## Ensure y is integers
    
    fir_names = []
    firs = []
    for j in range(X.shape[1]):
        
        tj = ts.TimeSeries(X[:,j], sampling_interval=tr)
        era = nta.EventRelatedAnalyzer(tj, ty, window)

        firs.append(era.FIR.data)
        fir_names.extend(ynames[1:])  ## Drop nan/baseline

    Xfir = np.vstack(firs).transpose()
    fir_names = np.asarray(fir_names)

    assert checkX(Xfir)
    assert Xfir.shape[0] == window, ("After FIR rows not equal to window")
    assert Xfir.shape[1] == (len(ynames[1:]) * X.shape[1]), ("After" 
        "FIR wrong number of features")
    assert fir_names.shape[0] == Xfir.shape[1], ("fir_names and Xfir" 
        "don't match")

    return Xfir, fir_names


def eva(X, y, trial_index, window, tr):
    """Average trials for each feature in X

    Parameters
     ----------
     X : 2D array-like (n_sample, n_feature)
         The data to decompose
     y : 1D array, None by default
         Sample labels for the data.  In y, np.nan and 'nan' values 
         are ignored.
     trial_index : 1D array (n_sample, )
         Each unique entry should match a trial.
     window : int 
         Trial length

     Return
     ------
     Xeva : a 2D arrays (n_feature*unique_y, window)
         The average trials
     feature_names : 1D array
         The names of the features (taken from y)
    """

    evas = []
    eva_names = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for j in range(X.shape[1]):
        Xtrials = []
        
        xj = X[:,j][:,np.newaxis]  ## Need 2D

        # Each feature into trials, rescale too
        Xtrial, feature_names = by_trial(xj, trial_index, window, y)
        Xtrial = scaler.fit_transform(Xtrial.astype(np.float))
        unique_fn = sorted(np.unique(feature_names))

        # and again by unique_y/fe]ature_names
        Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

        # put all that togthether
        Xtrials.extend([Xl.transpose() for Xl in Xlabels])

        # and average the trials then
        # name names.
        evas.extend([Xt.mean(axis=1) for Xt in Xtrials])
        eva_names.extend(unique_fn)

    # Reshape : (window, len(unique_y)*n_features)
    Xeva = np.vstack(evas).transpose()
    eva_names = np.asarray(eva_names)

    assert checkX(Xeva)
    assert Xeva.shape[0] == window, ("After EVA rows not equal to window")
    assert Xeva.shape[1] == len(unique_fn) * X.shape[1], ("After" 
        "EVA wrong number of features")
    assert eva_names.shape[0] == Xeva.shape[1], ("eva_names and Xeva" 
        "don't match")

    return Xeva, eva_names


if __name__ == '__main__':
    from wheelerdata.load.fh import FH 
    from fmrilearn.preprocess.labels import csv_to_targets
    from fmrilearn.load import load_meta
    from fmrilearn.load import load_nii
    from fmrilearn.preprocess.labels import filter_targets

    data = FH()
    
    metas = data.get_metapaths_containing('rt')
    targets = csv_to_targets(metas[0])

    paths = data.get_roi_data_paths('Insula')
    X = load_nii(paths[0], clean=True, sparse=False, smooth=False)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.astype(np.float))
    X = X[targets['TR'],:]
    X = X.mean(1)[:,np.newaxis]

    y = targets['rt']
    tc = targets['trialcount']
    Xfir, flfir = fir(X, y, tc, 20, 1.5)
    Xeva, fleva = eva(X, y, tc, 11, 1.5)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(Xeva)

    ax2 = fig.add_subplot(212)
    ax2.plot(Xfir[1:-1,1:])    

    fig.savefig("eva_fir compare.pdf")
