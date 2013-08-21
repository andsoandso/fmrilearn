"""Custom cross validation index generators."""
import numpy as np
from sklearn.cross_validation import KFold        

class SelectTargets(object):
    """Return indices to select, in order, each unique y_i in y.
    
    Parameters:
    ----------
    y - a list of targets.
    """
    def __init__(self, y, indices=True):
        self.y = np.array(y)
        self.uniquey = sorted(list(set(self.y.tolist())))
        self.n = len(self.uniquey)
        self.indices = indices
    
    
    def  __iter__(self):
        yindices = []
        for y_i in self.uniquey:
            # Create a mask for y_i and use it to
            # generate an index.        
            # sklearn.cross_validation.Classes()
            # return indices.
            mask = y_i == self.y
            if self.indices:
                yield np.arange(mask.shape[0])[mask]
            else:
                yield mask
    
            
    def __len__(self):
        return self.n
    

class KFoldChunks(object):
    """Create KFold style CV object, but operate on chunks (i.e contiguous
    groups) of labels instead of single labels.
     
    Parameters:
    ----------
    y - a list of targets.
    n_folds - number of folds.
    indices - if True, return lists of ints instead of booleans.
    min_size - The minimum chuck size (0 are ignored, > 1).
    """
    
    def __init__(self, y, n_folds, indices=True, min_size=1):
        self.labels = np.array(y)
        self.unique_labels = np.unique(y)
        
        self.n = self.labels.shape[0]
        self.n_unique_labels = self.unique_labels.shape[0]
        self.unique_index = np.arange(self.n_unique_labels)
        
        self.n_folds = int(n_folds)
        self.indices = indices
        self.min_size = int(min_size)
        
    
    def __iter__(self):
        fold_sizes = (self.n_unique_labels / self.n_folds) * np.ones(
                self.n_folds, dtype=np.int)
        fold_sizes[:self.n_unique_labels % self.n_folds] += 1
        
        current = 0
        if self.indices:
            ind = np.arange(self.n)        
        for ii, fold_size in enumerate(fold_sizes):
            
            # Select the fold using the unique set,
            unique_mask = np.zeros(self.n_unique_labels, dtype=np.bool)
            start, stop = current, current + fold_size
            unique_mask[self.unique_index[start:stop]] = True
            
            # then use the selected unique to fold
            # over the labels (i.e. y, the target).
            #
            # Init a mask as False, and flip when lab
            # from unique_index matches labels
            test_index = np.zeros(self.n, dtype=np.bool)
            for lab in self.unique_labels[unique_mask]:
                test_index[self.labels == lab] = True

            train_index = np.logical_not(test_index)
                ## And invert for train
            
            if self.indices:
                train_index = ind[train_index]
                test_index = ind[test_index]

            current = stop

            yield train_index, test_index    
                
            
    
    def __len__(self):
        return self.n_folds
    
