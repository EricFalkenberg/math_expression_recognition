from config import dataset_meta
import csv
import os
import progressbar
import numpy as np
import xml.etree.ElementTree as ET

def load_data(fname):
    with open(fname) as f:
        csvreader = csv.reader(f)
        X = []
        Y = []
        for row in csvreader:
            X.append(row[0])
            Y.append(row[1])
    return X, Y

def reposition_xy_points(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    for key, value in stroke_data.items():
        stroke_length = sum(sum([distance(s[x], s[x+1]) for x in range(len(s)-1)]) for s in value)
        print stroke_length
        print value
        break

def extract_xy_data(string):
    string = string.split(",")
    return [[float(j) for j in i.split()][:2] for i in string]

def retrieve_stroke_data(X, directory, config):
    trace_map = {}
    num_files = len(os.listdir(directory % config['location']))
    progress = progressbar.ProgressBar(max_value=num_files)
    curr = 0
    print "Processing files in %s" % (directory % config['location'])
    for filename in os.listdir(directory % config['location']):
        if filename not in config['exclude']:
            tree = ET.parse((directory+"%s") % (config['location'], filename))
            root = tree.getroot() 
            annotations = root.findall(config['xml_name_tag'])       
            loc = annotations[1].text
            trace = root.findall(config['xml_trace_tag'])
            trace_map[loc] = map(extract_xy_data, (stroke.text for stroke in trace))
        progress.update(curr)
        curr += 1
    progress.update(curr)
    return {i:trace_map[i] for i in X if i in trace_map}

def extract_features(fname):
    X, Y = load_data(fname)
    dirs = ["%s/trainingSymbols/", "%s/trainingJunk/"]
    for directory in dirs:
        stroke_data = retrieve_stroke_data(X, directory, dataset_meta) 
        reposition_xy_points(stroke_data)
            
        break # Just for now to speed things up

extract_features("tmp/real-test.csv")
