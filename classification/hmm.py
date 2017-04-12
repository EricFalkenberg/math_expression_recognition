import csv
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

    def __init__(this, X, Y, config, model_file=None):
        if model_file != None:
            with gzip.open(model_file) as f:
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
            if target in segmented_data:
                lengths = [55 for _ in range(len(segmented_data[target])/55)]
                model = model.fit(segmented_data[target])
                hmm_map[target] = model
                created += 1
            else:
                hmm_map[target] = None
                abstained += 1
            progress.update(curr)
            curr += 1
        print "%d/%d models created" % (created, created+abstained)
        this.models = hmm_map
        this.save_model("models/hmm.model")

    def save_model(this, fname):
        with gzip.open(fname, 'wb') as f:
            cPickle.dump(this.models, f, -1)

    def evaluate_model(this, samples, targets):
        print "Evaluating model"
        num_points = len(samples)/55 
        progress = progressbar.ProgressBar(max_value=num_points)
        curr = 0
        buff = []
        curr_target = None
        correct, incorrect = 0, 0
        with open('evals/hmm_out.csv', 'wb') as f:
            csvwriter = csv.writer(f)
            for idx, sample, target in zip(range(len(samples)), samples, targets):
                if idx % 55 == 0 and buff != []:
                    p_target = this.predict(buff)
                    if p_target == curr_target[1]:
                        correct += 1
                    else:
                        incorrect += 1 
                    csvwriter.writerow([curr_target[0], p_target])
                    buff = []
                    progress.update(curr)
                    curr += 1
                buff.append(sample)
                curr_target = target
        print "ACCURACY: %.2f" % (float(correct) / float(correct+incorrect))
        
             
    def predict(this, sample):
        best = None
        b_class = None
        score_class = [(model.score(sample, [55]), target) for target, model in this.models.items()]
        score_class = sorted(score_class)
        return [i[1] for i in score_class]
        #for target, model in this.models.items():
        #    if model == None:
        #        continue
        #    score = model.score(sample, [55])
        #    if score > best or best == None:
        #        best = score 
        #        b_class = target
        #return b_class
    
def from_model(fname):
    return classifier(None, None, None, fname)

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=hmm_meta['program_description'])
    parser.add_argument('command', **arg_command)
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    if args.command[0] == "train":
        gt, features = extract_features(args.dataset[0])
        c = classifier(features, gt, hmm_model)   
    elif args.command[0] == "test":
        c = from_model("models/hmm.model")
        gt, test_f   = extract_features(args.dataset[0])
        c.evaluate_model(test_f, gt)

