import argparse
import numpy as np
from hmmlearn import hmm
from config import hmm_meta, hmm_model, arg_dataset, arg_classes
from features import extract_features


class classifier:

    def __init__(this, X, Y, config):
        hmm_map = {}
        ## TODO: Split dataset up by class
        ## TODO: Train each model by class sample
        ## TODO: A lot of things
        for target in hmm_meta['class_names']:
            model = hmm.GMMHMM(**config)
            #model = model.fit(X)
            hmm_map[target] = model

    def evaluate_model(this, samples, targets):
        pass
             
    def predict(this, sample):
        pass
    

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=hmm_meta['program_description'])
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    gt, features = extract_features(args.dataset[0])
    c = classifier(features, gt, hmm_model)



