"""Functions for processing or using nifti data that are (somewhat at 
least) specific to the meta-accumulate project.

See my 'roi' package and the 'nibabel' library for more general methods.
"""
import re
import os
import roi
import nibabel as nb
import pandas as pd


def findallnii(datapath, match):
    """Find all the BOLD data in '$datapath/[...]/bold' directories 
    whose names partially <match>.
    """
    
    # ----
    # 1. Find all the functional (i.e. bold*) 
    # .nii data to process
    #
    # Loop over all subdirectories and files
    # adding functional data to datafiles
    datafiles = []
    for top, dirs, files in os.walk(datapath):
        # Only want to look in bold directories,
        # otherwise we waste a lot of time
        # looking in the .dcm and anatomical
        # directories.
        if re.search("bold", top):
            # Only keep files that match dataregex
            for name in files:       
                tmppath = os.path.join(top, name)
                if re.search(match, tmppath):
                    datafiles.append(tmppath)

    return datafiles
    

def masknii(mask, nii, save=None):
    """Apply the named <mask> to the named <nifti> data, 
    returning the masked data.  Save if <save> in not None.
    """
    
    mask_data = roi.atlas.get_roi('HarvardOxford', mask)
    if sum([nz.size for nz in mask_data.get_data().nonzero()]) == 0:
        raise ValueError("{0} was empty".format(mask))

    nii_data = roi.io.read_nifti(nii)
    if sum([nz.size for nz in nii_data.get_data().nonzero()]) == 0:
        raise ValueError("{0} was empty".format(nii))

    masked_nii_data = roi.pre.mask(nii_data, mask_data, standard=True)
                ## Even though we're in MNI152 the q_form
                ## for the fidl converted data is not set correctly
                ## the s_form is what the q should be
                ## thus standard=False
    
    if sum([nz.size for nz in masked_nii_data.get_data().nonzero()]) == 0:
        raise ValueError("{0} using {1} is empty".format(nii, mask))

    if save != None:
        print("Saving {0}.".format(save))
        roi.io.write_nifti(masked_nii_data, save)

    return masked_nii_data


def filternii(nii, meta, tr_col_name, save=None):
    """Filter out time-slices from nii that do not exists in the TR index
    in the supplied metadata file (.csv, column oriented).
    
    Parameters:
    ----------
    nii - a nifit1 file name
    meta - a csv of metadata
    tr_col_name - the name of the column in meta that contains the TRs
    save - If not None, file is saved as "save"
    
    Returns:
    -------
    The filtered nifti object.
    """
    
    nii_data = roi.io.read_nifti(nii)
    nii_array = nii_data.get_data()
        ## Gets the numpy array...
    
    meta = pd.read_csv(meta)
    trs = meta[tr_col_name].tolist()
    
    filtered_nii = nb.Nifti1Image(nii_array[:,:,:,trs], 
            nii_data.get_affine(), nii_data.get_header())
        
    if save != None:
        print("Saving {0}.".format(save))
        roi.io.write_nifti(filtered_nii, save)
    
    return filtered_nii


