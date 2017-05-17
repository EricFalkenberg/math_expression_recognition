import argparse
import os
from string import printable

from config import file_handler_config as fconfig
from file_handler import read_training_data

from classification import random_forest
from classification.config import random_forest_meta as rnd_forest_config
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

    def evaluate_model(this):
        for path in this.dataset:
            f_handler  = this.dataset[path]
            if not f_handler.is_malformed():
                ## Get predicted output
                traces     = f_handler.traces
                store_name, _ = os.path.splitext(os.path.basename(path))
                with open("Baseline_output/{0}.lg".format(store_name), 'w') as f:
                    for tid in traces:
                        f.write(str(s_object('2', tid)))
                    s_object.reset()
                ## Get ground truth
                #groups     = f_handler.groups
                #with open("Gt_output/{0}.lg".format(store_name), 'w') as f:
                #    for group in groups:
                #        ids = ', '.join(str(i) for i in group.traces_id)
                #        print group.truth, group.type, group.id
                #        f.write(str(s_object(group.type, ids)))

if __name__ == '__main__':
    dataset = read_training_data(fconfig['training_data_loc'])
    s = segmenter(dataset, "classification/models/random_forest.model")
    s.evaluate_model()

