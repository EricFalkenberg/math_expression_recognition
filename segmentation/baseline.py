import argparse
from string import ascii_letters

from config import file_handler_config as fconfig
from file_handler import read_training_data

from classification import random_forest
from classification.config import random_forest_meta as rnd_forest_config
import classification.features as cl_features




class s_object:
    __slots__ = ('label', 'modifier', 'strokes') 
    label_gen = { cl : (n for n in ascii_letters) for cl in rnd_forest_config['class_names'] }

    def __init__(this, label, strokes):
        this.label    = label
        this.modifier = s_object.label_gen[label].next()
        this.strokes  = strokes
    
    def __str__(this):
        return "O, {0}_{1}, 1.0, {2}".format(this.label, this.modifier, this.strokes)

class segmenter:
    __slots__ = ('dataset', 'classifier')
   
    def __init__(this, dataset, classifier_path):
        this.classifier = random_forest.classifier(None, None, None, classifier_path) 
        this.dataset = dataset

    def evaluate_model(this):
        for path in this.dataset:
            f_handler = this.dataset[path]
            traces    = f_handler.traces
            for tid in traces:
                print s_object('2', tid)
            break
        

if __name__ == '__main__':
    dataset = read_training_data(fconfig['training_data_loc'])
    s = segmenter(dataset, "classification/models/random_forest.model")
    s.evaluate_model()

