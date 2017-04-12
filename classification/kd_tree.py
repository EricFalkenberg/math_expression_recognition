import csv
import argparse
import gzip
import cPickle
from features import extract_features
from config import kdtree_meta, kdtree_model, arg_command, arg_dataset, arg_classes
from sklearn.neighbors import KDTree
import numpy as np

class classifier:

    def __init__(this, X, Y, config, model_file=None):
        if model_file != None:
            with gzip.open(model_file) as f:
                this.tree, this.Y = cPickle.load(f)
        else:
            this.Y = Y
            this.tree = KDTree(X, **config)
            this.save_model("models/kd_tree.model")

    def save_model(this, fname):
        with gzip.open(fname, 'wb') as f:
            cPickle.dump([this.tree, this.Y], f, -1)

    def evaluate_model(this, samples, targets):
        correct, incorrect = 0, 0
        with open("evals/kd_tree.eval", "wb") as f:
            csvwriter = csv.writer(f)
            for sample, target in zip(samples, targets):
                p = this.predict(sample)
                if p[1] == target[1]: correct += 1
                else: incorrect += 1
                csvwriter.writerow([target[0], p[1]])
        print "CORRECT PERCENTAGE: %.2f" % (float(correct) / (correct+incorrect))
             
    
    def predict(this, sample):
        dist, ind = this.tree.query([sample], k=1)
        return this.Y[ind[0,0]]

def from_model(fname):
    return classifier(None, None, None, fname)

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=kdtree_meta['program_description']) 
    parser.add_argument('command',  **arg_command)
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()
    ## Run classifier
    if args.command[0] == "train":
        gt, features = extract_features(args.dataset[0])
        c = classifier(features, gt, kdtree_model)   
    elif args.command[0] == "test":
        c = from_model("models/kd_tree.model")
        gt, test_f   = extract_features(args.dataset[0])
        c.evaluate_model(test_f, gt)
