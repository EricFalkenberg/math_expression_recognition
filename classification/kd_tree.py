import argparse
from features import extract_features
from config import kdtree_meta, kdtree_model, arg_dataset, arg_classes
from sklearn.neighbors import KDTree
import numpy as np

class classifier:

    def __init__(this, X, Y, config):
        this.X = X
        this.Y = Y
        this.tree = KDTree(X, **config)
    
    def predict(this, sample):
        dist, ind = this.tree.query(sample, k=1)
        print dist, ind
        print this.Y[ind[0,0]]

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=kdtree_meta['program_description']) 
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()
    ## Run classifier
    gt, features = extract_features(args.dataset[0])
    c = classifier(features, gt, kdtree_model)   
    gt, test_f   = extract_features("tmp/tmp.csv")
    c.predict(test_f[0]) 
