from string import printable
import os
import argparse
from file_handler import read_training_data, relation
from config import file_handler_config as fconfig
from config import baseline_meta, arg_data_type
from classification.config import random_forest_meta as rnd_forest_config

class s_object:
    __slots__ = ('label', 'truth', 'modifier', 'strokes') 

    label_gen = { cl : (n for n in printable[1:]) for cl in rnd_forest_config['class_names'] }

    @staticmethod
    def reset():
        s_object.label_gen = { 
            cl : (n for n in printable[1:]) for cl in rnd_forest_config['class_names'] 
        }

    def __init__(this, label, truth, strokes):
        this.label    = label
        this.truth    = truth
        this.modifier = s_object.label_gen[label].next()
        this.strokes  = strokes
    
    def __str__(this):
        return "O, {1}, {0}, 1.0, {2}\n".format(this.label, this.truth, this.strokes)

class parser:
    
    def __init__(this, dataset):
        pass

    def evaluate_model(this, out_name):
        try:
            os.mkdir(out_name)
        except OSError as e:
            print "[WARNING]: Directory {0} already exists".format(out_name)
        for path in dataset:
            store_name, _ = os.path.splitext(os.path.basename(path))
            with open("{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                f_handler = dataset[path]
                groups    = f_handler.groups
                for group in groups:
                    ids = ', '.join(str(i) for i in group.traces_id)
                    f.write(str(s_object(group.type, group.truth, ids)))
                f.write('\n')
                for group1, group2 in [(groups[i], groups[i+1]) for i in range(0,len(groups)-1)]:
                    f.write(str(relation(group1.truth, group2.truth, "Right")))
                                 
                

    def output_ground_truth(this, out_name):
        try:
            os.mkdir("ground_truth/{0}".format(out_name))
        except OSError as e:
            print "[WARNING]: Directory ground_truth/{0} already exists".format(out_name)
        for path in dataset:
            store_name, _ = os.path.splitext(os.path.basename(path))
            with open("ground_truth/{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                f_handler = dataset[path]
                relations = f_handler.relationship_graph.relations
                groups    = f_handler.groups
                for group in groups:
                    ids = ', '.join(str(i) for i in group.traces_id)
                    f.write(str(s_object(group.type, group.truth, ids)))
                f.write('\n')
                for r in relations:
                    f.write(str(r))
        

    

if __name__ == '__main__':
    ## Parse command line arguments
    p = argparse.ArgumentParser(description=baseline_meta['program_description']) 
    p.add_argument('data_type',  **arg_data_type)

    args = p.parse_args()
    data_type = args.data_type[0]

    dataset = read_training_data(fconfig['training_data_{0}'.format(data_type)])
    s = parser(dataset)
    s.evaluate_model("parser_{0}".format(data_type))
    s.output_ground_truth("parser_{0}".format(data_type))

    #s = parser(dataset, "classification/models/random_forest.model")
    #s.evaluate_model("test_{0}".format(data_type))
