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
    
    checkX(Xeva)
    if Xeva.shape[0] != window:
        raise ValueError("After EVA rows not equal to window")
    if Xeva.shape[1] != ((len(unique_y) + 1) * X.shape[1]):
        print(X.shape)
        print(len(unique_y)+1)
        print(Xeva.shape)

        raise ValueError("After EVA wrong number of features")
    if eva_names.shape[0] != Xeva.shape[1]:
        raise ValueError("eva_names and Xeva don't match")

    return Xeva, feature_names


# def _eva_X(X, trial_index, window):
    
#     X = X.astype(float)
    
#     M = None  ## Will be a 2d array
#     yM = []
    
#     # Create a list of trial mask
#     trial_masks = []
#     for trial in np.unique(trial_index):
#         trial_masks.append(trial == trial_index)
    
#     # Iterate over the masks, averaging as we go.
#     k = 1.0
#     for mask in trial_masks:
#         # Get the trials data
#         Xtrial = X[mask,] 
#         Xtrial = Xtrial[0:window,]
        
#         # and norm to first TR
#         first_tr = Xtrial[0,:].copy().astype(float)
#         for i in range(Xtrial.shape[0]):
#             Xtrial[i,:] = Xtrial[i,:] / first_tr
        
#         # then update the mean.
#         if k == 1:
#             M = Xtrial.copy()
#         else:
#             Mtmp = M.copy()
#             M += (Xtrial - Mtmp) / k;
#                 ## Online mean taken from the online var method of
#                 ## B. P. Welford and is presented in Donald Knuth's Art of 
#                 ## Computer Programming, Vol 2, page 232, 3rd edition. 
#                 ##
#                 ## Math:
#                 ## Mk is the online mean, k is the sample index (:= 1)
#                 ## xk is the kth sample
#                 ## Mk = Mk-1 + (xk - Mk-1)/k 
#                 ## Sk = Sk-1 + (xk - Mk-1)*(xk - Mk).
#                 ## For 2 <= k <= n, the kth estimate of the variance is s^2 = 
#                 ## Sk/(k - 1).
#         k += 1.0
    
#     return M


# def eva(X, y, trial_index, window):
#     """Estimate the average trial response for each feature (column) in
#     X for each condition in y."""
    
#     # ---
#     # Setup
#     checkX(X)
#     nrow, ncol = X.shape
    
#     Xeva = None
#     yeva = None
    
#     # ----
#     # Remove short trials...
#     # Find them
#     locations = locate_short_trials(trial_index, window)
    
#     # Build up a mask and apply it
#     # removing the short trials
#     if len(locations) > 0:
#         short_mask = locations.pop() == trial_index
#         for i in locations:
#             short_mask = short_mask | (i == trial_index)
#         short_mask = np.logical_not(short_mask)
    
#         X = X[short_mask,]
#         y = y[short_mask]
#         trial_index = trial_index[short_mask]
                
#     # ----
#     # For each cond calc the EVA...
#     conds = np.unique(y)
#     for i, cond in enumerate(conds):
#         mask = cond == y
#         Xtmp = _eva_X(X[mask,], trial_index[mask], window)
#         ytmp = np.repeat(cond, window)
#         index_tmp = np.arange(ytmp.shape[0])
        
#         if i == 0:
#             Xeva = Xtmp.copy()
#             yeva = ytmp.copy()
#             timecourse_index = index_tmp.copy()
#         else:
#             Xeva = np.vstack([Xeva, Xtmp])
#             yeva = np.concatenate([yeva, ytmp])
#             timecourse_index = np.concatenate([timecourse_index, index_tmp])
    
#     checkX(Xeva)
#     if Xeva.shape[0] != yeva.shape[0]:
#         raise ValueError("After eva y doesn't match X")
#     if Xeva.shape[0] != timecourse_index.shape[0]:
#         raise ValueError("After eva timecourse_index doesn't match X")
#     if X.shape[1] != Xeva.shape[1]:
#         raise ValueError("After eva columns don't match")

#     return Xeva, yeva, timecourse_index


# def fir(X, y, tr, window):
#     """Return a FIR estimate of each event.
    
#     DO NOT USE - CALC COMPLETES BUT IS INCORRECT!
    
#     Parameters
#     ---------
#     x - a 1/2d array of data (col oriented if 2d)
#         eventlabs - a set of trial-level events
#     y - a 1d array of labels
#     tr - the sampling rate (repition time if BOLD data)
#         window - the length of the trial (in tr).
#     window - the trial length/estimate HRF duration
    
#     Return
#     -----
#     Xfir - a 2d array where cols are voxels (matching X) and rows are
#         the trial mean for each cond concatenated
#     yfir - a 1d array of labels matching each row in Xmean to a cond
#     """

#     checkX(X)
#     nrow, ncol = X.shape
#     conds = np.unique(y)

#     # Truncate X or y as needed.
#     if len(y) < nrow:
#         X = X[0:len(y),:]
#     elif len(y) > nrow:
#         y = y[0:nrow]

#     # Setup Xfir, a 2d array of trial estimates
#     Xfir = np.zeros((len(conds)*window, ncol))
#     for j in range(ncol):
#         # Get the col
#         x = np.array(X[:,j]).squeeze()

#         # Cast to a nitime object, and use nitime to calc the FIR
#         # cond estimates.
#         tx = ts.TimeSeries(x, sampling_interval=tr)
#         ty = ts.TimeSeries(y, sampling_interval=tr)
#         era = nta.EventRelatedAnalyzer(tx, ty, window)
#         import pdb; pdb.set_trace()

#         Xfir[:,j] = era.FIR.data.flatten()
#             ## want a 1d array with each conds 
#             ## estimate concatentated, so flatten.
    
#     yfir = _create_cond_y(conds, window)

#     return Xfir, yfir
