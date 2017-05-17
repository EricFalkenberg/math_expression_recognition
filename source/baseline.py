import argparse
import os
from string import printable

from config import file_handler_config as fconfig
from config import baseline_meta, arg_data_type
from file_handler import read_training_data

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
    __slots__ = ('dataset', 'classifier')
   
    def __init__(this, dataset, classifier_path):
        this.classifier = random_forest.classifier(None, None, None, classifier_path) 
        this.dataset = dataset

    def evaluate_model(this, out_name):
        try:
            os.mkdir(out_name)
        except OSError as e:
            print "[WARNING]: Directory {0} already exists".format(out_name)
        for path in this.dataset:
            f_handler  = this.dataset[path]
            if not f_handler.is_malformed():
                ## Get predicted output
                traces     = f_handler.traces
                store_name, _ = os.path.splitext(os.path.basename(path))
                with open("{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                    for tid in traces:
                        features   = extract_features_from_sample([traces[tid].data])
                        p = this.classifier.predict(features)
                        f.write(str(s_object(p, tid)))
                    s_object.reset()

if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=baseline_meta['program_description']) 
    parser.add_argument('data_type',  **arg_data_type)

    args = parser.parse_args()
    data_type = args.data_type[0]

    dataset = read_training_data(fconfig['training_data_{0}'.format(data_type)])
    s = segmenter(dataset, "classification/models/random_forest.model")
    s.evaluate_model("test_{0}".format(data_type))

