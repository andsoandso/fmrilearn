import numpy as np

from fmrilearn.info import print_X_info
from fmrilearn.info import print_label_counts
from fmrilearn.save import _addcol
from fmrilearn.preprocess.labels import locate_short_trials
from fmrilearn.preprocess.data import checkX


def restack(X, feature_names):
    """Reshape X into a stack of matrices based on feature names, 
    one new 'layer' for each unique (sorted) entry in x.

    Note
    ----
    In order to ensure the layers are stackable, if the number of cols 
    is off zero are added as pad whereever needed.
    """

    X = np.array(X)
    feature_names = np.array(feature_names)
    unique_names = np.unique(feature_names)

    nrow = X.shape[0]
    if X.shape[1] != feature_names.shape[0]:
        raise ValueError("Number of features in X doesn't match feature_names.")
   
    # Init the reshaped X (Xstack) and the feature
    # mask, then loop over the rest
    mask = unique_names[0] == feature_names 
    assert mask.shape[0] == feature_names.shape[0], ("The mask was the" 
        "wrong shape")
    assert np.sum(mask) > 1, ("The mask was empty")

    Xstack = X[:,mask]
    for name in unique_names[1:]:
        mask = name == feature_names
        assert np.sum(mask) > 1, ("The mask was empty")

        Xname = X[:,mask]
        diff = Xstack.shape[1] - Xname.shape[1]
        if diff < 0:
            Xstack = _addcol(Xstack, np.abs(diff))
        elif diff > 0:
            Xcond = _addcol(Xname, np.abs(diff))
        Xstack = np.vstack([Xstack, Xname])

    fn_stack = []
    for name in unique_names:
        fn_stack.extend([name, ] * nrow)
    fn_stack = np.array(fn_stack)

    assert checkX(Xstack)
    assert fn_stack.shape[0] == Xstack.shape[0], ("After stacking X and" 
        "feature_names did not match.")

    return Xstack, fn_stack


def by_trial(X, trial_index, window, y):
    """Rehapes X so each trial is feature. If y is not None, y
    is converted to feature_names array.

    Note
    ----
    In y, np.nan and 'nan' values are ignored.
    """

    ncol = X.shape[1]

    # ----
    # Remove short trials from X.
    locations = locate_short_trials(trial_index, window)
    if len(locations) > 0:
        short_mask = locations.pop() == trial_index
        for i in locations:
            short_mask = short_mask | (i == trial_index)
        short_mask = np.logical_not(short_mask)
    
        X = X[short_mask,]
        trial_index = trial_index[short_mask]

    # ----
    # Find all the trials
    trial_masks = []
    for trial in np.unique(trial_index):
        if np.isnan(trial): continue
        trial_masks.append(trial == trial_index)
    
    # And split up X
    Xlist = []
    feature_names = []
    for mask in trial_masks:
        y0 = y[mask][0]
        if np.str(y0) != 'nan':
            Xlist.append(X[mask,][0:window,])
            feature_names.append(np.repeat(y0, ncol))

    feature_names = np.hstack(feature_names)

    # Create Xtrial by horizonal stacking
    Xtrial = np.hstack(Xlist)

    # Sanity
    assert checkX(Xtrial)
    assert Xtrial.shape[1] == feature_names.shape[0], ("After reshape" 
        "Xtrial and feature_names don't match")
    assert Xtrial.shape[0] == window, ("Number of samples in Xtrial" 
        "doesn't match window")
    
    return Xtrial, feature_names
