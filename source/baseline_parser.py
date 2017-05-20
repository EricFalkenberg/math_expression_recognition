from string import printable
from matplotlib import pyplot as plt
from random import random
import networkx as nx
import os
import argparse
from file_handler import read_training_data, relation
from config import file_handler_config as fconfig
from config import baseline_meta, arg_data_type, arg_command
from classification.config import random_forest_meta as rnd_forest_config
from classification.features import smooth_xy_points, reposition_xy_points
from features import normalize_coords, has_los, create_image_from_points

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

    def create_relation_graph(this, groups):
        G = nx.DiGraph()
        G.add_node('base')
        for g in groups:
            G.add_node(g.truth)
            G.add_edge('base', g.truth, weight=0)
        group_trace_lens = [len(g.traces) for g in groups]
        combined_strokes = []
        for group in groups:
            traces = [i.data for i in group.traces]
            smooth = smooth_xy_points({'id':traces})
            reposi = reposition_xy_points(smooth)['id']
            combined_strokes.extend(reposi)
        img    = create_image_from_points(combined_strokes)
        groups_data = []
        group_trace_lens = group_trace_lens
        idx = 0
        f = 0
        while idx < len(group_trace_lens):
            s = f
            f = s+group_trace_lens[idx]
            groups_data.append(combined_strokes[s:f])
            idx += 1
        for group1, idx1 in zip(groups_data, range(len(groups_data))):
            for group2, idx2 in zip(groups_data, range(len(groups_data))):
                if idx1 != idx2:
                    los = False
                    for stroke1 in group1:
                        for stroke2 in group2:
                            for p1 in stroke1:
                                for p2 in stroke2:
                                    los = has_los(p1, p2, img)
                                    if los:
                                        G.add_edge(groups[idx1].truth, groups[idx2].truth, \
                                                   weight=random(), label='test')
                                        break
                                if los:
                                    break
                            if los:
                                break
                        if los:
                            break
        #print "GRAPH"
        #print G.nodes()
        #for u,v in G.edges():
        #    print u,v,G.get_edge_data(u,v)
        #nx.draw_networkx(G, arrows=True, with_labels=True, node_size=600)
        #plt.show()
        edmonds = nx.algorithms.tree.Edmonds(G)
        g = edmonds.find_optimum(kind='max')
        #print "GRAPH AFTER EDMONDS"
        #print g.nodes()
        #for u,v in g.edges():
        #    print u,v,g.get_edge_data(u,v)
        return g

if __name__ == '__main__':
    ## Parse command line arguments
    p = argparse.ArgumentParser(description=baseline_meta['program_description']) 
    p.add_argument('command', **arg_command)
    p.add_argument('data_type',  **arg_data_type)

    args = p.parse_args()
    data_type = args.data_type[0]

    dataset = read_training_data(fconfig['training_data_{0}'.format(data_type)])
    s = parser(dataset)
    #s.evaluate_model("parser_{0}".format(data_type))
    #s.output_ground_truth("parser_{0}".format(data_type))

    #s = parser(dataset, "classification/models/random_forest.model")
    #s.evaluate_model("test_{0}".format(data_type))
    for i in dataset:
        f_handler = dataset[i]
        print i
        if not f_handler.is_malformed():
            s.create_relation_graph(f_handler.groups)
        print
        print
