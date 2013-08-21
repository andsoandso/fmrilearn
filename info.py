def print_target_info(targets):
    for name, labels in targets.items():
        print("\tNumber of {0}: {1}".format(name, labels.shape))

def print_X_info(X):
    print("\tX shape: {0}".format(X.shape))

def print_clf_info(clf):
    print("\tClassifying with {0}".format(clf))
    
    