"""Functions for saving fmrilearn data and results."""
import os
import csv
import pandas as pd
import numpy as np
from fmrilearn.load import load_tcdf
from fmrilearn.load import decompose_tcdf


def _process_meta(meta):
    # Convert meta if it's a seq
    if isinstance(meta, str):
        metastr = meta
    else:
        metastr = ",".join([str(m) for m in meta])

    return metastr


def save_accuracy_table(name, meta, acc, concatenate=True):
    if not(concatenate):
        reset_table(name)
    
    metastr = _process_meta(meta)

    f = open(name,'a')
    f.write("{0},{1}\n".format(metastr, acc))
    f.close()


def reset_accuracy_table(name):
    """Deprecated, use reset_table()"""
    reset_table(name)


def reset_table(name):
    try:
        os.remove(name)
    except OSError:
        pass

def _do_resize(name, X):
    """Compares the number of columns in X and the named tcdf file.

    Parameters
    ----------
    name: csv tcdf-like
        Then name of a tcdf data table (.csv)
    X: 2D array-like
        A (n_sample, n_feature) data set

    Return
    ------
    do_resize: boolean (default False)
        Should we resize X or the named
    diff: int (default 0)
        What was the difference between name and X column counts
        (i.e. ncol_name - ncol_X)
    tcdf: None or DataFrame (default None)
        If resize is True, return the tcdf, otherwise
        return None
    
    Note
    ----
    The tcdf DataFrame must *only* contain the data, and three
    metadata cols.  Bad things happen if there is other metadata 
    present.
    """
    
    do_resize = False
    diff = 0
    tcdf = None

    try:
        tcdf = load_tcdf(name)
    except Exception:
        return do_resize, diff, tcdf
    
    diff = (tcdf.shape[1] - 3) - X.shape[1] 
        ## 3 adjusts for tcdf metadata

    if diff != 0: 
        do_resize = True
    
    return do_resize, diff, tcdf


def _addcol(X, ncol):
    """Add ncol to X

    Parameters
    ----------
    X: 2D array-like
        A (n_sample, n_feature) data set
    ncol: int 
        How many cols to add
    """
        
    if ncol < 0:
        raise ValueError("ncol must be positive")

    ncol = int(ncol)
    
    i, j = X.shape
    Xnew = np.zeros([i, j+ncol], dtype=X.dtype)
    Xnew[:,0:j] = X

    return Xnew


def save_tcdf(name, X, cond, dataname, index='auto', header=True, mode='w',
        float_format=None):
    """Save X and the provided metadata as a 'tcdf' style table.
    
    Parameters
    ----------
    TODO

    Note
    ----
    For more info on the tcdf format see:
        TODO github link to boldplotr
    """
    
    if mode == 'w':
        reset_table(name)

    # Process metadata args
    # If they were sting-like,
    # expand into arrays.
    # And create index if 'auto'.
    nrow = X.shape[0]
    if hasattr(cond, "capitalize"):
        cond = np.repeat(cond, nrow)
    if hasattr(dataname, "capitalize"):
        dataname = np.repeat(dataname, nrow)
    if index == 'auto':
        index = np.arange(nrow)
    else:
        index = np.array(index)

    # Should we do any resizing?
    resize, diff, tcdf_old = _do_resize(name, X)
    if resize:
        header = True
        mode = 'w'  ## If resizing we don't want to append want to overwrite
                    ## while looking like an append to the user.

        X_old, cond_old, dataname_old, index_old = decompose_tcdf(tcdf_old)
        if diff < 0:
            X_old = _addcol(X_old, np.abs(diff))
        elif diff > 0:
            X = _addcol(X, np.abs(diff))            
        else:
            raise ValueError("Resize was True, but diff was 0")

        # Combined that new and old data
        X = np.vstack([X_old, X])
        cond = np.concatenate([cond_old, cond])
        dataname = np.concatenate([dataname_old, dataname])
        index = np.concatenate([index_old, index])

        # Few quick sanity checks
        if X.shape[0] != cond.shape[0]:
            raise ValueError("After resize X and cond length mismatch")
        if X.shape[0] != dataname.shape[0]:
            raise ValueError("After resize X and dataname lengths mismatch")
        if X.shape[0] != index.shape[0]:
            raise ValueError("After resize X and index lengths mismatch")
 
    tcdf = pd.DataFrame(X)
    tcdf["cond"] = cond
    tcdf["dataname"] = dataname
    tcdf["index"] = index
    tcdf.to_csv(name, 
            sep=",", mode=mode, header=header, 
            index=False, float_format=float_format)

