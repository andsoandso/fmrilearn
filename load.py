"""Functions for loading fMRI data."""
import numpy as np
import nibabel as nb
import pandas as pd

from scipy.sparse import csc_matrix

from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.data import smooth as smoothfn
from fmrilearn.preprocess.data import remove_invariant_features


def load_roifile(roifile):
    """Parse and return rois and names found in the roifile.
    
    Parameters:
    ----------
    roifile - a file containing names of ROIs to use (see below).
    
    Format:
    ------
    roifiles have a limited and rigid format (no header, quotes, comments, 
    etc):
    
        roi:name
        
    Where 'roi' is the name of a region of interest in the HarvardOxford
    atlas (see my roi package for details) and name is a name for that roi.
    ....The HarvardOxford names are rather long.  
    
    For example:
        Middle Frontal Gyrus:mfg
        Insula:insula
    
    Note: roifile parsing has no sanity checks.  It assumes formatting is
        perfect.
    """
    
    f = open(roifile, "r")
    
    rois = []
    names = []
    for row in f:
        roi, name = row.split(":")
        
        roi = roi.strip()
        name = name.strip()
            ## Don't want trialing 
            ## whitespace and newlines
        
        rois.append(roi)
        names.append(name)

    f.close()    

    return rois, names


def load_meta(conds, meta):
    """Return the metadata matching."""
    
    # Ensure cond is a seq
    if not hasattr(conds, "__getitem__"):
        raise TypeError("Conds is not a sequence")
    
    metadf = pd.read_csv(meta)
    ys = []
    for cond in conds:
        try:
            y = np.array(metadf[cond].tolist())
        except KeyError:
            raise KeyError("{0} not found in {1}".format(cond, meta))
        ys.append(y)
    
    return ys


def load_nii(nifiti, clean=True, sparse=False, smooth=False, **kwargs):
    """Convert the nifiti-1 file into a 2D array (n_sample x n_features).
    
    Parameters
    ----------
    nifti - str
        The name of the data to load
    clean - boolean (True)
        Remove invariant features features?  If used n_features will 
        not match n_voxels in the orignal nifit1 file.  This operation
        is not reversable.  If you clean there is probablity little
        point in converting to a sparse representation.
    sparse - boolean (False)
        Use the (CSC) sparse format (True)?
    smooth - boolean (False)
        High/low pass filter the data?
    [, ...] - Optional parameters for smooth 
        (defaults: tr=1.5, ub=0.06, lb=0.006)

    Return
    ------
    X - 2D array (n_sample x n_features)
        The BOLD data
    """
    
    # Data is 4d (x,y,z,t) we want 2d, where each column is 
    # a voxel and each row is the temporal (t) data
    # i.e. the final shape should be (x*y*x, t)
    nii = nb.nifti1.load(nifiti)

    numt = nii.shape[3]
    numxyz = nii.shape[0] * nii.shape[1] * nii.shape[2]
    dims = (numxyz, numt)
    
    # Get into 2d (n_feature, n_sample)
    X = nii.get_data().astype('int16').reshape(dims).transpose()
    if clean:
        X = remove_invariant_features(X, sparse=False)
    
    if smooth:
        # Setup smooth params
        tr = 1.5
        ub = 0.06
        lb = 0.001
        if "tr" in kwargs:
            tr = kwargs["tr"]
        if "ub" in kwargs:
            ub = kwargs["ub"]
        if "lb" in kwargs:
            ub = kwargs["lb"]
        
        X = smoothfn(X, tr=tr, ub=ub, lb=lb)
    
    assert checkX(X)
    
    if sparse: 
        X = csc_matrix(X)

    return X


def decompose_tcdf(tcdf):
    """Decompose tcdf into its parts - X, cond, dataname, index."""

    index = np.array(tcdf["index"].tolist())
    cond = np.array(tcdf["cond"].tolist())
    dataname = np.array(tcdf["dataname"].tolist())
    
    tcdf = tcdf.drop(labels=["index", "cond", "dataname"], axis=1)
    X = np.array(tcdf.as_matrix())

    assert checkX(X)
    
    return X, cond, dataname, index


def load_tcdf(name):
    """Loads the named tcdf file

    Note
    ----
    For more on the tcdf format see TODO
    """

    return pd.read_csv(name)
