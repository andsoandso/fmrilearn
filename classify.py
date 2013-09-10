import numpy as np
from sklearn.preprocessing import scale
from fmrilearn.preprocess.labels import create_y
from fmrilearn.info import print_label_counts


def simple(Xtrain, Xtest, labels_train, labels_test, clf, verbose):
    """Run a very simple classification exp. 
    
    Parameters
    ----------
    X - a 2d array, column oriented
    labels - a list of class labels, one for each row in X
    clf - a sklearn classifier (that implements .fit() and 
        .predict())
    verbose - Print out useful debugging/status info (True).  If False
        this function is silent.

    Returns 
    -------
    truths - a list of correct classes for each test set
    predictions - a list of the predicted classes for each test set.
    """

    Xtest = np.array(Xtest)
    Xtrain = np.array(Xtrain)
    
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)    
    
    ytrain = create_y(labels_train)
    ytest = create_y(labels_test)
        ## Labels as ints
    
    if verbose:
        print("\tTrain labels:")
        print_label_counts(labels_train)
        print("\tTest labels:")
        print_label_counts(labels_test)
        
        print("\tShapes of Xtrain and ytrain: {0}, {1}".format(
                Xtrain.shape, ytrain.shape))
        print("\tNumber of Xtest and ytest: {0}, {1}".format(
                Xtest.shape, ytest.shape))
    
    clf.fit(scale(Xtrain), ytrain)

    predictions = clf.predict(scale(Xtest))
    truths = ytest
    
    return truths, predictions
    

def simpleCV(X, labels, cv, clf, verbose=True):
    """Run a simple CV based classification exp. 
    
    Parameters
    ----------
    X - a 2d array, column oriented
    labels - a list of class labels, one for each row in X
    cv - a sklearn cross-validation object
    clf - a sklearn classifier (that implements .fit() and 
        .predict())
    verbose - Print out useful debugging/status info (True).  If False
        this function is silent.

    Returns 
    -------
    truths - a list of correct classes for each test set
    predictions - a list of the predicted classes for each test set.
    """
    
    X = np.array(X)
    labels = np.array(labels)
    
    y = create_y(labels)
        ## Labels as ints
    
    truths = []
    predictions = []
    for train_index, test_index in cv:
        # ----
        # Partition the data
        Xtrain = X[train_index,:]
        Xtest = X[test_index,:]        
        ytrain = y[train_index]
        ytest = y[test_index]
        
        if verbose:
            print("Next fold:")
            print("\tShapes of Xtrain and Xtest: {0}, {1}".format(
                    Xtrain.shape, Xtest.shape))
            print("\tNumber of ytrain and ytest: {0}, {1}".format(
                    ytrain.shape, ytest.shape))

        # ----
        # Class!
        clf.fit(scale(Xtrain), ytrain)

        truths.append(ytest)
        predictions.append(clf.predict(scale(Xtest)))
    
    return truths, predictions

