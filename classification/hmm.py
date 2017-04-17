import logging
import progressbar
logging.captureWarnings(True)
import argparse
import numpy as np
import cPickle
import gzip
from hmmlearn import hmm
from config import hmm_meta, hmm_model, arg_command, arg_dataset, arg_classes
from features import extract_features


class classifier:

    def __init__(this, X, Y, config, model_name=None):
        if model_name != None:
            with gzip.open(model_name) as f:
                this.models = cPickle.load(f)
            return
        segmented_data = {}
        for target, feature in zip(Y, X):
            t = target[1]
            if t not in segmented_data:
                segmented_data[t] = []
            segmented_data[t].append(feature)

        hmm_map = {}
        num_points = len(hmm_meta['class_names']) 
        progress = progressbar.ProgressBar(max_value=num_points)
        curr = 0
        created = 0
        abstained = 0

        print "Training HMM Class Models"
        for target in hmm_meta['class_names']:
            model = hmm.GMMHMM(**config)
            if target in segmented_data :
                lengths = [55 for _ in range(len(segmented_data[target])/55)]
                model = model.fit(segmented_data[target])
                hmm_map[target] = model
                created += 1
            else:
                hmm_map[target] = None
                abstained += 1
            progress.update(curr)
            curr += 1
        this.models = hmm_map

        print "%d/%d models created" % (created, created+abstained)
        with gzip.open("models/hmm.model", 'wb') as f:
            cPickle.dump(this.models, f, -1)

    def evaluate_model(this, samples, targets):
        buff = []
        curr_target = None
        correct, incorrect = 0, 0
        for idx, sample, target in zip(range(len(samples)), samples, targets):
            if idx % 55 == 0 and buff != []:
                p_target = this.predict(buff)
                if p_target == curr_target:
                    correct += 1
                else:
                    incorrect += 1 
                buff = []
            buff.append(sample)
            curr_target = target[1]
        print "CLASSIFICATION: %.4f" % (float(correct) / float(correct+incorrect))
        
             
    def predict(this, sample):
        best = -1
        b_class = None
        for target, model in this.models.items():
            if model == None:
                continue
            score = model.score(sample, [55])
            if score > best:
                best = score
                b_class = target
        return b_class
    

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=hmm_meta['program_description'])
    parser.add_argument('command', **arg_command)
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    cmd     = args.command[0]
    dataset = args.dataset[0]
    
    if cmd == "train":
        gt, features = extract_features(args.dataset[0], time_series=True)
        c = classifier(features, gt, hmm_model)
    if cmd == "test":
        c = classifier(None, None, None, "models/hmm.model") 
        c.evaluate_model(features, gt)
