"""Functions to analyze features."""
import numpy as np
import nitime.timeseries as ts
import nitime.analysis as nta

from copy import deepcopy
from scipy.stats  import spearmanr, pearsonr

from fmrilearn.preprocess.labels import locate_short_trials
from fmrilearn.info import print_X_info, print_label_counts


def _checkX(X):
    """Is X OK to use?"""

    if X.ndim != 2:
        raise ValueError("X must be 2d.")


def _split_1darray_by_trials(x, trial_index, max_trial_length):
    trialcodes = np.unique(trial_index)
    trialarray = np.zeros((max_trial_length, len(trialcodes)))
    for j, code in enumerate(trialcodes):
        trialmask = code == trial_index
        trialarray[0:np.sum(trialmask),j] = x[trialmask]
    
    return trialarray


def _create_cond_y(conds, trial_length):
    # Finally create the cond labels
    y = []
    for cond in conds:
        y.extend([cond, ] * trial_length)

    return np.array(y)


def correlate(X, y, corr="spearman"):
    """Correlate each feature in X, with y (some meaningful 
    set of dummmy coded labels).
     
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
    
    _checkX(X)

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


def _eva_X(X, trial_index, window):
    
    X = X.astype(float)
    
    M = None  ## Will be a 2d array
    yM = []
    
    # Create a list of trial mask
    trial_masks = []
    for trial in np.unique(trial_index):
        trial_masks.append(trial == trial_index)
    
    # Iterate over the masks, averaging as we go.
    k = 1.0
    for mask in trial_masks:
        # Get the trials data
        Xtrial = X[mask,] 
        Xtrial = Xtrial[0:window,]
        
        # and norm to first TR
        first_tr = Xtrial[0,:].copy().astype(float)
        for i in range(Xtrial.shape[0]):
            Xtrial[i,:] = Xtrial[i,:] / first_tr
        
        # then update the mean.
        if k == 1:
            M = Xtrial.copy()
        else:
            Mtmp = M.copy()
            M += (Xtrial - Mtmp) / k;
                ## Online mean taken from the online var method of
                ## B. P. Welford and is presented in Donald Knuth's Art of 
                ## Computer Programming, Vol 2, page 232, 3rd edition. 
                ##
                ## Math:
                ## Mk is the online mean, k is the sample index (:= 1)
                ## xk is the kth sample
                ## Mk = Mk-1 + (xk - Mk-1)/k 
                ## Sk = Sk-1 + (xk - Mk-1)*(xk - Mk).
                ## For 2 <= k <= n, the kth estimate of the variance is s^2 = 
                ## Sk/(k - 1).
        k += 1.0
    
    return M


def eva(X, y, trial_index, window, verbose=True):
    """Estimate the average trial response for each feature (column) in
    X for each condition in y."""
    
    # ---
    # Setup
    X = np.array(X)
    y = np.array(y)
    trial_index = np.array(trial_index)
    
    _checkX(X)
    nrow, ncol = X.shape
    
    Xeva = None
    yeva = None
    
    # ----
    # Remove short trials...
    # Find them
    locations = locate_short_trials(trial_index, window)
    
    if verbose:
        print_label_counts(y)
        print_X_info(X)
        print("Examining trial lengths")
        print("\t{0} trials detected".format(np.unique(trial_index).shape[0]))
        print("\t{0} short trial(s) detected".format(len(locations)))
        
    # Build up a mask
    if len(locations) > 0:
        short_mask = locations.pop() == trial_index
        for i in locations:
            short_mask = short_mask | (i == trial_index)
        short_mask = np.logical_not(short_mask)
    
        # Apply the mask
        X = X[short_mask,]
        y = y[short_mask]
        trial_index = trial_index[short_mask]
    
    # ----
    # For each cond calc the EVA...
    conds = np.unique(y)
    for i, cond in enumerate(conds):
        mask = cond == y
        Xtmp = _eva_X(X[mask,], trial_index[mask], window)
        ytmp = np.repeat(cond, window)
        index_tmp = np.arange(ytmp.shape[0])
        
        if i == 0:
            Xeva = Xtmp.copy()
            yeva = ytmp.copy()
            timecourse_index = index_tmp.copy()
        else:
            Xeva = np.vstack([Xeva, Xtmp])
            yeva = np.concatenate([yeva, ytmp])
            timecourse_index = np.concatenate([timecourse_index, index_tmp])
    
    if verbose:
        print("\tEVA complete")
        print_X_info(Xeva)
        
    return Xeva, yeva, timecourse_index


def fir(X, y, tr, window):
    """Return a FIR estimate of each event.
    
    Parameters
    ---------
    x - a 1/2d array of data (col oriented if 2d)
        eventlabs - a set of trial-level events
    y - a 1d array of labels
    tr - the sampling rate (repition time if BOLD data)
        window - the length of the trial (in tr).
    window - the trial length/estimate HRF duration
    
    Return
    -----
    Xfir - a 2d array where cols are voxels (matching X) and rows are
        the trial mean for each cond concatenated
    yfir - a 1d array of labels matching each row in Xmean to a cond
    """

    _checkX(X)
    nrow, ncol = X.shape
    conds = np.unique(y)

    # Truncate X or y as needed.
    if len(y) < nrow:
        X = X[0:len(y),:]
    elif len(y) > nrow:
        y = y[0:nrow]

    # Setup Xfir, a 2d array of trial estimates
    Xfir = np.zeros((len(conds)*window, ncol))
    for j in range(ncol):
        # Get the col
        x = np.array(X[:,j]).squeeze()

        # Cast to a nitime object, and use nitime to calc the FIR
        # cond estimates.
        tx = ts.TimeSeries(x, sampling_interval=tr)
        ty = ts.TimeSeries(y, sampling_interval=tr)
        era = nta.EventRelatedAnalyzer(tx, ty, window)
        import pdb; pdb.set_trace()

        Xfir[:,j] = era.FIR.data.flatten()
            ## want a 1d array with each conds 
            ## estimate concatentated, so flatten.
    
    yfir = _create_cond_y(conds, window)

    return Xfir, yfir
