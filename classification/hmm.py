import argparse
import numpy as np
from hmmlearn import hmm
from config import hmm_meta, hmm_model, arg_dataset, arg_classes
from features import extract_features


class classifier:

    def __init__(this, X, Y, config):
        hmm_map = {}
        for c in hmm_meta['class_names']:
            model = hmm.GMMHMM(**hmm_model)
            hmm_map[c] = model
            #gt, features = extract_features(args.dataset[0])
            #trainModel(model, features)

    def evaluate_model(this, samples, targets):
        correct, incorrect = 0, 0
        for sample, target in zip(samples, targets):
            p = this.predict(sample)
            if p[1] == target[1]: correct += 1
            else: incorrect += 1
        print "CORRECT PERCENTAGE: %.2f" % (float(correct) / (correct+incorrect))
             
    def predict(this, sample):
        hmm.predict(sample)
        return hmm
    

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=hmm_meta['program_description'])
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    c = classifier(None, None, None)



