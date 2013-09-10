"""Functions for saving fmrilearn data and results."""
import os

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

