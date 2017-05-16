import logging
import progressbar
logging.captureWarnings(True)
import argparse
import numpy as np
import cPickle
import gzip
from sklearn.ensemble import RandomForestClassifier
from config import random_forest_meta, random_forest_model, arg_command, arg_dataset, arg_classes
from features import extract_features


class classifier:

    def __init__(this, X, Y, config, model_name=None):
        class_num_map = dict()
        idx = 0
        for c in random_forest_meta['class_names']:
            if c not in class_num_map:
                class_num_map[c] = idx
                idx += 1
        this.class_num_map = class_num_map
        this.num_class_map = { v:k for k,v in class_num_map.items() }

        if model_name != None:
            with gzip.open(model_name) as f:
                this.model = cPickle.load(f)
            return

        model = RandomForestClassifier(n_estimators=50)
        y = []
        for target in Y:
            y.append(class_num_map[target[1]])
        
        model.fit(X, y)

        with gzip.open("models/random_forest.model", 'wb') as f:
            cPickle.dump(model, f, -1)

    def evaluate_model(this, samples, targets):
        predictions = this.model.predict(samples)
        for t, p in zip(targets, predictions):
            print "%s, %s" % (t[0], this.num_class_map[p])
             

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=random_forest_meta['program_description'])
    parser.add_argument('command', **arg_command)
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    cmd     = args.command[0]
    dataset = args.dataset[0]
    
    if cmd == "train":
        gt, features = extract_features(args.dataset[0])
        c = classifier(features, gt, random_forest_model)
    if cmd == "test":
        c = classifier(None, None, None, "models/random_forest.model") 
        gt, features = extract_features(args.dataset[0])
        c.evaluate_model(features, gt)
