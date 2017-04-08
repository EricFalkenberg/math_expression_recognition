import argparse
from config import kdtree_meta, kdtree_model, arg_dataset, arg_classes
from sklearn.neighbors import KDTree
import numpy as np

class classifier:

    def __init__(this, X, Y, config):
        this.X = X
        this.Y = Y
        this.tree = KDTree(X, **config)

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=kdtree_meta['program_description']) 
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()
    ## Run classifier
    c = classifier(np.random.random((10, 2)), np.random.randint(0, 1), kdtree_model)
