import argparse
from config import kdtree_config
from sklearn.neighbors import KDTree
import numpy as np

class classifier:

    def __init__(this, X, Y):
        this.X = X
        this.Y = Y
        this.tree = KDTree(X)

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=kdtree_config['description']) 
    args = parser.parse_args()
    ## Run classifier
    c = classifier(np.random.random((10, 2)), np.random.randint(0, 1))
