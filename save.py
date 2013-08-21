"""Functions for saving fmrilearn data and results."""
import os

def save_accuracy_table(name, meta, acc, concatenate=True):
    if not(concatenate):
        reset_accuracy_table(name)
    
    f = open(name,'a')
    
    # Convert meta if it's a seq
    if isinstance(meta, str):
        metastr = meta
    else:
        metastr = ",".join([str(m) for m in meta])
    
    f.write("{0},{1}\n".format(metastr, acc))
    f.close()


def reset_accuracy_table(name):
    try:
        os.remove(name)
    except OSError:
        pass

