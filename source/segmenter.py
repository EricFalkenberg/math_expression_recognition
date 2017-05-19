import argparse
import gzip
import cPickle
import os
from string import printable

from config import file_handler_config as fconfig
from config import baseline_meta, arg_command, arg_data_type
from file_handler import read_training_data, split_data
from features import msscf, stroke_symbol_pair_features, preprocess_strokes

from sklearn.ensemble import AdaBoostClassifier

from classification import random_forest
from classification.config import random_forest_meta as rnd_forest_config
from classification.features import extract_features_from_sample
import classification.features as cl_features

class s_object:
    __slots__ = ('label', 'modifier', 'strokes') 

    label_gen = { cl : (n for n in printable) for cl in rnd_forest_config['class_names'] }

    @staticmethod
    def reset():
        s_object.label_gen = { 
            cl : (n for n in printable) for cl in rnd_forest_config['class_names'] 
        }

    def __init__(this, label, strokes):
        this.label    = label
        this.modifier = s_object.label_gen[label].next()
        this.strokes  = strokes
    
    def __str__(this):
        return "O, {0}_{1}, {0}, 1.0, {2}\n".format(this.label, this.modifier, this.strokes)

class segmenter:
    __slots__ = ('train_names', 'test_names', 'dataset', 'classifier')
   
    def __init__(this, train_names, test_names, dataset, classifier_path, model_name=None):
        if model_name != None:
            with gzip.open(model_name) as f:
                this.classifier, this.dataset, this.train_names, this.test_names,\
                this.model = cPickle.load(f)
            return
        this.classifier = random_forest.classifier(None, None, None, classifier_path) 
        this.dataset = dataset
        this.train_names = train_names
        this.test_names = test_names
        this.model = AdaBoostClassifier(n_estimators=25)
        this.train()
        with gzip.open("models/adaboost_segmenter.model", 'wb') as f:
            cPickle.dump([this.classifier, this.dataset, this.train_names, this.test_names,\
                          this.model], f, -1)
        

    def train(this):
        X = []
        y = []
        for sample_name in this.train_names:
            f_handler = this.dataset[sample_name]    
            if not f_handler.is_malformed():
                strokes   = sorted([v for k,v in f_handler.traces.items()], key=lambda x: x.id)
                groups    = f_handler.groups
                join_map  = {}
                for group in groups:
                    sorted_ids = sorted(group.traces_id)
                    join_map.update({ sorted_ids[i]:sorted_ids[i+1] for i in range(len(sorted_ids)-1) })
                for s1, s2 in [[strokes[i],strokes[i+1]] for i in range(len(strokes)-1)]:
                    to_join = join_map[s1.id] == s2.id if s1.id in join_map else False
                    ## GET FEATURES
                    s1f, s2f           = preprocess_strokes([s1, s2]) 
                    shape_context      = msscf(s1f, s2f)
                    geometric_features = stroke_symbol_pair_features(s1f, s2f)
                    features   = extract_features_from_sample([s1f, s2f])
                    class_probs_join   = this.classifier.model.predict_proba(features)
                    features   = extract_features_from_sample([s1f, s2f])
                    class_probs_sep    = this.classifier.model.predict_proba(features)
                    sample = []
                    sample.extend(shape_context)
                    sample.extend(geometric_features)
                    sample.extend(class_probs_join[0].T)
                    sample.extend(class_probs_sep[0].T)
                    X.append(sample)
                    y.append(1 if to_join else 0)
        this.model.fit(X,y)

    def evaluate_model(this, out_name):
        try:
            os.mkdir(out_name)
        except OSError as e:
            print "[WARNING]: Directory {0} already exists".format(out_name)
        try:
            os.mkdir("ground_truth/{0}/".format(out_name))
        except OSError as e:
            print "[WARNING]: Directory ground_truth/{0} already exists".format(out_name)

        for path in this.test_names:
            f_handler  = this.dataset[path]
            if not f_handler.is_malformed():
                strokes   = sorted([v for k,v in f_handler.traces.items()], key=lambda x: x.id)
                output    = [[strokes[0]]]
                for s1, s2 in [[strokes[i],strokes[i+1]] for i in range(len(strokes)-1)]:
                    ## GET FEATURES
                    s1f, s2f           = preprocess_strokes([s1, s2]) 
                    shape_context      = msscf(s1f, s2f)
                    geometric_features = stroke_symbol_pair_features(s1f, s2f)
                    features   = extract_features_from_sample([s1f, s2f])
                    class_probs_join   = this.classifier.model.predict_proba(features)
                    features   = extract_features_from_sample([s1f, s2f])
                    class_probs_sep    = this.classifier.model.predict_proba(features)
                    sample = []
                    sample.extend(shape_context)
                    sample.extend(geometric_features)
                    sample.extend(class_probs_join[0].T)
                    sample.extend(class_probs_sep[0].T)
                    to_join = this.model.predict(sample)
                    if to_join:
                        output[-1].append(s2)
                    else:
                        output.append([s2])
                ## Get predicted output
                #traces     = f_handler.traces
                store_name, _ = os.path.splitext(os.path.basename(path))
                with open("{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                    for group in output:
                        data_array = [i.data for i in group]   
                        ids = ", ".join([str(i.id) for i in group])
                        features   = extract_features_from_sample(data_array)
                        prediction = this.classifier.predict(features)
                        f.write(str(s_object(prediction, ids)))
                    s_object.reset()
                groups     = f_handler.groups
                with open("ground_truth/{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                    for group in groups:
                        ids = ', '.join(str(i) for i in group.traces_id)
                        f.write(str(s_object(group.type, ids)))

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=baseline_meta['program_description']) 
    parser.add_argument('command',  **arg_command)
    parser.add_argument('data_type',  **arg_data_type)

    args = parser.parse_args()
    data_type = args.data_type[0]
    ## Run classifier
    if args.command[0] == "train":
        dataset = read_training_data(fconfig['training_data_{0}'.format(data_type)])
        train_names, test_names = split_data(dataset, 2.0/3.0)
        s = segmenter(train_names, test_names, dataset, "classification/models/random_forest.model")
    elif args.command[0] == "test":
        s = segmenter(None, None, None, None, "models/adaboost_segmenter.model")
        s.evaluate_model("test_{0}".format(data_type))
