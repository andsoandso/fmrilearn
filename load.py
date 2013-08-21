"""Functions for loading fMRI data."""
import numpy as np
import nibabel as nb
from scipy.sparse import csc_matrix


def load_nii(nifiti, sparse=True):
    """Return nifiti data in nifti format and convert it to a 2d array, 
    in a example X feature (voxel) format.
    
    Parameters
    ----------
    nifti - the name of the data to load
    sparse - use the (CSC) sparse format (True)?
    """
    
    # Data is 4d (x,y,z,t) we want 2d, where each column is 
    # a voxel and each row is the temporal (t) data
    # i.e. the final shape should be (x*y*x, t)
    nii = nb.nifti1.load(nifiti)

    numt = nii.shape[3]
    numxyz = nii.shape[0] * nii.shape[1] * nii.shape[2]
    dims = (numxyz, numt)
    
    # Get into 2d then sparse-ify?  Transpose once
    # we're sparse
    X = nii.get_data().astype('int16').reshape(dims).transpose()
    if sparse: X = csc_matrix(X)

    return X
