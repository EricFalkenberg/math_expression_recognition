import os
import time
import math
from copy import copy
import random
from bs4 import BeautifulSoup as Soup
import logging
from classification.config import random_forest_meta
from config import file_handler_config
logging.captureWarnings(True)

class trace:
    __slots__ = ('id', 'data')
    
    def __init__(this, t):
        this.id = int(t['id'])
        this.data = this.format_data(t.string)

    def format_data(this, s):
        s = s.split(",")
        s = [i.split() for i in s]
        return [[float(i[0]), float(i[1])] for i in s]

class group:
    __slots__ = ('id', 'type', 'truth', 'traces')
    
    def __init__(this, g, trace_data):
        this.id = g['xml:id']
        this.type = g.annotation.string
        this.truth = g.annotationxml['href']
        this.traces = [int(i['tracedataref']) for i in g.find_all('traceview')]
        this.traces_id = [int(i['tracedataref']) for i in g.find_all('traceview')]
        this.map_traces(trace_data)
        this.fix_info()

    def fix_info(this):
        if this.type == ",":
            this.type  = "COMMA"
            this.truth = "COMMA{0}".format(this.truth[1:]) 

    def map_traces(this, trace_data):
        this.traces = map(lambda x: trace_data[x], this.traces)
        
class relation:
    __slots__ = ('parent', 'child', 'type')

    def __init__(this, parent, child, rtype):
        this.parent = parent
        this.child = child
        this.type = rtype

    def __str__(this):
        return "R, {0}, {1}, {2}, 1.0\n".format(this.parent, this.child, this.type)

class relation_graph:
    __slots__ = ('root', 'relations')
    
    types = ['undefined', 'right', 'subscript', 'superscript', 'above', 'below', 'inside']

    def __init__(this, root):
        this.relations = []
        this.root = this.define_graph(root.findChildren(recursive=False)[0])

    def define_graph(this, curr_node):
        if curr_node.name == 'mrow':
            p, c = curr_node.findChildren(recursive=False)
            p_final = this.define_graph(p)
            c_final = this.define_graph(c)
            this.relations.append(relation(p_final, c_final, 'Right'))
            return p_final
        if curr_node.name == 'msub':
            p, c = curr_node.findChildren(recursive=False)
            p_final = this.define_graph(p)
            c_final = this.define_graph(c)
            this.relations.append(relation(p_final, c_final, 'Subscript'))
            return p_final
        if curr_node.name == 'msup': 
            p, c = curr_node.findChildren(recursive=False)
            p_final = this.define_graph(p)
            c_final = this.define_graph(c)
            this.relations.append(relation(p_final, c_final, 'Superscript'))
            return p_final
        if curr_node.name == 'msubsup':
            p, c1, c2 = curr_node.findChildren(recursive=False)
            p_final = this.define_graph(p)
            c1_final = this.define_graph(c1)
            c2_final = this.define_graph(c2)
            this.relations.append(relation(p_final, c1_final, 'Subscript'))
            this.relations.append(relation(p_final, c2_final, 'Superscript'))
            return p_final
        if curr_node.name in ['mi', 'mn', 'mo']:
            return curr_node['xml:id']
        raise Exception("None of blocks triggered\n{0}".format(curr_node.prettify()))

class f_handler:
    __slots__ = ('groups', 'traces', 'relationship_graph')
    
    def __init__(this, groups, traces, rel_graph):
        this.groups = groups
        this.traces = traces
        this.relationship_graph = rel_graph

    def num_classes(this):
        classes = {}
        for group in this.groups:
            if group.type not in classes:
                classes[group.type] = 0
            classes[group.type] += 1
        return classes

    def is_malformed(this):
        return this.groups == None or this.traces == None or \
               len(this.groups) == 0 or len(this.traces) == 0

    def __str__(this):
        return "f_handler(num_groups={0}, total_num_traces={1})"\
                .format(len(this.groups), len(this.traces))

def read_inkml(fname, include_relations=False):
    with open(fname) as f:
        soup = Soup(f)
        try:
            traces    = { t.id: t for t in [trace(i) for i in soup.find_all('trace')] }
            groups    = [group(i, traces) for i in soup.tracegroup.find_all('tracegroup')]
            rel_graph = relation_graph(soup.annotationxml.math)
            return groups, traces, rel_graph
        except Exception as e:
            print e.message
            raise Exception
            return None, None, None

def read_directory(directory):
    out_map = {}
    files = os.listdir(directory)
    directory = directory+"/{}"
    for fn in files:
        fname = directory.format(fn)
        _, ext = os.path.splitext(fname)
        if ext == ".inkml":
            groups, traces, rel_graph = read_inkml(fname)
            out_map[fname] = f_handler(groups, traces, rel_graph)  
    return out_map

def read_training_data(dir_names):
    f_handlers = {}
    for dir_name in dir_names: 
         f_handlers.update(read_directory(dir_name))
    return f_handlers

def add_to_dict(old, new):
    old = copy(old)
    for k,v in new.items():
        old[k] += v
    return old

def sub_from_dict(old, new):
    old = copy(old)
    for k,v in new.items():
        old[k] -= v
    return old

def dict_error(d1, d2, t1, t2):
    error = 0
    for k,v in d2.items():
        if v+d1[k] > 0:
            error += abs(t2-(v/(v+d1[k])))
    return error

def split_data(f_handlers, train_percentage):
    train_names = []
    test_names  = []
    f_names     = [i for i in f_handlers if not f_handlers[i].is_malformed()]
    train_nums  = { k:0 for k in random_forest_meta['class_names'] }
    test_nums   = { k:0 for k in random_forest_meta['class_names'] }

    remaining = range(len(f_names))
    upper_bound = len(f_names)-1
    while float(len(remaining))/len(f_names) > train_percentage:
        chosen_idx = random.randint(0, upper_bound)
        chosen_num = remaining[chosen_idx] 
        remaining  = remaining[:chosen_idx]+remaining[chosen_idx+1:]
        test_names.append(f_names[chosen_num])    
        test_nums = add_to_dict(test_nums, f_handlers[f_names[chosen_num]].num_classes()) 
        upper_bound -= 1

    for ridx in remaining:
        train_names.append(f_names[ridx])
        train_nums = add_to_dict(train_nums, f_handlers[f_names[ridx]].num_classes())

    for _ in range(100000):
        train_idx  = random.randint(0, len(train_names)-1)
        train_name = train_names[train_idx]        
        curr_err   = dict_error(train_nums, test_nums, train_percentage, 1.0-train_percentage)
        step_train = sub_from_dict(train_nums, f_handlers[train_name].num_classes())
        step_test  = add_to_dict(test_nums, f_handlers[train_name].num_classes()) 
        step_err   = dict_error(step_train, step_test, train_percentage, 1.0-train_percentage)
        if step_err < curr_err:
            train_nums = step_train
            test_nums  = step_test
            test_names.append(train_name)
            train_names = train_names[:train_idx]+train_names[train_idx+1:]

        test_idx   = random.randint(0, len(test_names)-1)
        test_name  = test_names[test_idx]
        curr_err   = dict_error(train_nums, test_nums, train_percentage, 1.0-train_percentage)
        step_test  = sub_from_dict(test_nums, f_handlers[test_name].num_classes())
        step_train = add_to_dict(train_nums, f_handlers[test_name].num_classes())
        step_err   = dict_error(step_train, step_test, train_percentage, 1.0-train_percentage)
        if step_err < curr_err:
            train_nums = step_train
            test_nums  = step_test
            train_names.append(test_name)
            test_names = test_names[:test_idx]+test_names[test_idx+1:]

    return train_names, test_names                

if __name__ == '__main__':
    f = file_handler_config['training_data_tiny']
    read_training_data(f)
