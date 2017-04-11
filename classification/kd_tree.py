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

    def evaluate_model(this, samples, targets):
        correct, incorrect = 0, 0
        for sample, target in zip(samples, targets):
            p = this.predict(sample)
            if p == target[1]: correct += 1
            else: incorrect += 1
        print "CORRECT PERCENTAGE: %.2f" % (correct / (correct+incorrect))
             
    
    def predict(this, sample):
        dist, ind = this.tree.query(sample, k=1)
        return this.Y[ind[0,0]][1]

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=kdtree_meta['program_description']) 
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()
    ## Run classifier
    gt, features = extract_features(args.dataset[0])
    c = classifier(features, gt, kdtree_model)   
    gt, test_f   = extract_features("tmp/test.csv")
    c.evaluate_model(test_f, gt)
