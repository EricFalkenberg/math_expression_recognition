import segmenter
import random
import numpy as np
from string import printable
import gzip
import cPickle
from matplotlib import pyplot as plt
import networkx as nx
import os
import argparse
from file_handler import read_training_data, relation, split_data
from config import file_handler_config as fconfig
from config import parser_meta, arg_data_type, arg_command
from classification.config import random_forest_meta as rnd_forest_config
from classification.features import smooth_xy_points, reposition_xy_points
from features import normalize_coords, has_los, create_image_from_points, calculate_bounding_box
from features import preprocess_strokes, msscf, stroke_symbol_pair_features
from sklearn.ensemble import RandomForestClassifier

class s_object:
    __slots__ = ('label', 'truth', 'modifier', 'strokes', 'test_subset') 

    def __init__(this, label, truth, strokes):
        this.label    = label
        this.truth    = truth
        this.strokes  = strokes
    
    def __str__(this):
        return "O, {1}, {0}, 1.0, {2}\n".format(this.label, this.truth, this.strokes)

class parser:
    __slots__=('train_names', 'test_name', 'dataset', 'segmenter_path')   

    class_num_map = { 'Right' : 0, 'Subscript' : 1, 'Superscript' : 2,
                      'Above' : 3, 'Below' : 4, 'Inside' : 5, 'None' : 6 }
    names_map = ['Right', 'Subscript', 'Superscript', 'Above', 'Below', 'Inside', 'None']
 
    def __init__(this, train_names, test_names, dataset, segmenter_path, model_name=None):
        if model_name != None:
            with gzip.open(model_name) as f:
                this.segmenter, this.dataset, this.train_names, this.test_names,\
                this.model = cPickle.load(f)
            return

        this.segmenter = None#segmenter.segmenter(None, None, None, None, segmenter_path) 
        this.dataset = dataset
        this.train_names = train_names
        this.test_names = test_names
        this.model = RandomForestClassifier(n_estimators=50) 
        this.train()
        print "STORING MODEL in models/rf_parser.model"
        with gzip.open("models/rf_parser.model", 'wb') as f:
            cPickle.dump([this.segmenter, this.dataset, this.train_names, this.test_names,\
                          this.model], f, -1)

    def train(this):
        print "TRAINING PARSER"
        X = []
        y = []
        for sample_name in this.train_names:
            f_handler = this.dataset[sample_name]    
            if not f_handler.is_malformed():
                trace_map = { g.truth:g.traces for g in f_handler.groups}
                for rel in f_handler.relationship_graph.relations:
                    group1_traces = trace_map[rel.parent]
                    group2_traces = trace_map[rel.child]
                    both_traces = []
                    both_traces.extend(group1_traces)
                    both_traces.extend(group2_traces)
                    proc_traces = preprocess_strokes(both_traces)
                    if proc_traces == None:
                        continue
                    group1_traces = proc_traces[:len(group1_traces)]
                    group2_traces = proc_traces[len(group1_traces):] 
                    bbox1  = sum([np.array(calculate_bounding_box(i)) for i in \
                                 group1_traces])/len(group1_traces)
                    bbox2  = sum([np.array(calculate_bounding_box(i)) for i in \
                                 group2_traces])/len(group2_traces)
                    center1  = (bbox1[0]+bbox1[1])/2., (bbox1[2]+bbox1[3])/2.
                    center2  = (bbox2[0]+bbox2[1])/2., (bbox2[2]+bbox2[3])/2.
                    a_center = (center1[0]+center2[0])/2, (center1[1]+center2[1])/2
                    combined1 = []
                    for stroke in group1_traces:
                        combined1.extend(stroke)
                    parent_shape_context = msscf(combined1, [], center=a_center)
                    combined2 = []
                    for stroke in group2_traces:
                        combined2.extend(stroke)
                    child_shape_context  = msscf(combined2, [], center=a_center)
                    geometric_features   = stroke_symbol_pair_features(combined1, combined2)
                    sample = []
                    sample.extend(parent_shape_context)
                    sample.extend(child_shape_context)
                    sample.extend(geometric_features) 
                    X.append(sample)
                    y.append(parser.class_num_map[rel.type])
        this.model.fit(X,y)
        r, w = 0, 0
        for sample,target in zip(X, y):
            p = this.model.predict(sample)
            if p == target:
                r += 1
            else:
                w += 1
        print "{0} Accuracy on Training Data".format(float(r)/float(r+w))

    def evaluate_model(this, out_name, num_eval, use_segmenter=False):
        print "EVALUATING MODEL"
        if use_segmenter:
            print "LOADING SEGMENTER"
            this.segmenter = segmenter.segmenter(None, None, None, None, \
                                                'models/adaboost_segmenter.model') 
            this.segmenter.dataset = None
            this.segmenter.train_names = None
            this.segmenter.test_names  = None
            print "FINISHED"
        this.test_subset = set()
        while len(this.test_subset) < num_eval:
            idx = random.randint(0, len(this.test_names)-1)
            name = this.test_names[idx]
            this.test_subset.add(name)
        try:
            os.mkdir(out_name)
        except OSError as e:
            print "[WARNING]: Directory {0} already exists".format(out_name)
        for path in this.test_subset:
            if use_segmenter:
                store_name, _ = os.path.splitext(os.path.basename(path))    
                with open("{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                    groups = segmenter.evaluate_single(path)
                    g = s.create_relation_graph(groups)
                    for group in groups:
                        ids = ', '.join(str(i) for i in group.traces_id)
                        f.write(str(s_object(group.type, group.truth, ids)))
                    f.write('\n')
                    for u,v in g.edges():
                        f.write(str(relation(u, v, g[u][v]['label'])))
            else:
                f_handler = this.dataset[path]
                if not f_handler.is_malformed():
                    store_name, _ = os.path.splitext(os.path.basename(path))    
                    print "EVALUATING: {0}".format(store_name)
                    if len(f_handler.groups) < 20:
                        g = s.create_relation_graph(f_handler.groups)
                        with open("{0}/{1}.lg".format(out_name, store_name), 'w') as f:
                            groups    = f_handler.groups
                            for group in groups:
                                ids = ', '.join(str(i) for i in group.traces_id)
                                f.write(str(s_object(group.type, group.truth, ids)))
                            f.write('\n')
                            for u,v in g.edges():
                                f.write(str(relation(u, v, g[u][v]['label'])))

    def output_ground_truth(this, out_name):
        #try:
        #    os.mkdir("ground_truth/{0}".format(out_name))
        #except OSError as e:
        #    print "[WARNING]: Directory ground_truth/{0} already exists".format(out_name)
        #for path in this.test_subset:
        #    store_name, _ = os.path.splitext(os.path.basename(path))
        #    with open("ground_truth/{0}/{1}.lg".format(out_name, store_name), 'w') as f:
        #        f_handler = this.dataset[path]
        #        relations = f_handler.relationship_graph.relations
        #        groups    = f_handler.groups
        #        for group in groups:
        #            ids = ', '.join(str(i) for i in group.traces_id)
        #            f.write(str(s_object(group.type, group.truth, ids)))
        #        f.write('\n')
        #        for r in relations:
        #            f.write(str(r))
        for path in this.test_subset:
            print path

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
                                        group1_traces = group1 
                                        group2_traces = group2
                                        both_traces = []
                                        both_traces.extend(group1_traces)
                                        both_traces.extend(group2_traces)
                                        proc_traces = preprocess_strokes(both_traces, raw=True)
                                        if proc_traces == None:
                                            continue
                                        group1_traces = proc_traces[:len(group1_traces)]
                                        group2_traces = proc_traces[len(group1_traces):] 
                                        bbox1  = sum([np.array(calculate_bounding_box(i)) for i in \
                                                     group1_traces])/len(group1_traces)
                                        bbox2  = sum([np.array(calculate_bounding_box(i)) for i in \
                                                     group2_traces])/len(group2_traces)
                                        center1  = (bbox1[0]+bbox1[1])/2., (bbox1[2]+bbox1[3])/2.
                                        center2  = (bbox2[0]+bbox2[1])/2., (bbox2[2]+bbox2[3])/2.
                                        a_center = (center1[0]+center2[0])/2, (center1[1]+center2[1])/2
                                        combined1 = []
                                        for stroke in group1_traces:
                                            combined1.extend(stroke)
                                        parent_shape_context = msscf(combined1, [], center=a_center)
                                        combined2 = []
                                        for stroke in group2_traces:
                                            combined2.extend(stroke)
                                        child_shape_context  = msscf(combined2, [], center=a_center)
                                        geometric_features   = stroke_symbol_pair_features(combined1, combined2)
                                        sample = []
                                        sample.extend(parent_shape_context)
                                        sample.extend(child_shape_context)
                                        sample.extend(geometric_features) 
                                        ws = this.model.predict_proba(sample)[0]
                                        weight,label = max( \
                                                    [(w,l) for (w,l) in sorted(zip(ws, parser.names_map))])
                                        G.add_edge(groups[idx1].truth, groups[idx2].truth, \
                                                   weight=weight, label=label)
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
        for u,v in g.edges():
            g[u][v]['label'] = G[u][v]['label']
        #nx.draw_networkx(g, arrows=True, with_labels=True, node_size=600)
        #plt.show()
        #print "GRAPH AFTER EDMONDS"
        #print g.nodes()
        #for u,v in g.edges():
        #    print u,v,G.get_edge_data(u,v)
        return g

if __name__ == '__main__':
    ## Parse command line arguments
    p = argparse.ArgumentParser(description=parser_meta['program_description']) 
    p.add_argument('command', **arg_command)
    p.add_argument('data_type',  **arg_data_type)

    args = p.parse_args()
    data_type = args.data_type[0]
    command = args.command[0]

    if command == "train":
        print "READING DATASET"
        dataset = read_training_data(fconfig['training_data_{0}'.format(data_type)])
        print "SPLITTING INTO TRAIN/TEST"
        train_names, test_names = split_data(dataset, 2.0/3.0)
        s = parser(train_names, test_names, dataset, "models/adaboost_segmenter.model")
    elif command == "test":
        s = parser(None, None, None, None, "models/rf_parser.model")
        s.evaluate_model("test_parser_{0}".format(data_type), 100)
        s.output_ground_truth("test_parser_{0}".format(data_type))
    elif command == "test_segmenter":
        s = parser(None, None, None, None, "models/rf_parser.model")
        s.evaluate_model("test_parser_{0}".format(data_type), 100, use_segmenter=True)
        s.output_ground_truth("test_parser_{0}".format(data_type))
