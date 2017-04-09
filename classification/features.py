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
        segments_length = [sum([distance(s[x], s[x+1]) for x in range(len(s)-1)]) for s in value]
        stroke_length = sum(segments_length)
        percentages = [float(i)/stroke_length if stroke_length != 0 else 0 for i in segments_length]
        processed = []
        for stroke, percentage, length in zip(value, percentages, segments_length):
            points_to_place = np.round(percentage*45)
            if points_to_place == 0:
                continue
            points_placed = 1
            repositioned = [stroke[0]]
            len_separate = float(length) / points_to_place
            length_to_next_point = len_separate
            index = 0
            while points_placed < points_to_place:
                while length_to_next_point > distance(stroke[index], stroke[index+1]):
                    length_to_next_point -= distance(stroke[index], stroke[index+1])
                    index += 1
                if stroke[index+1][0]-stroke[index][0] != 0:
                    theta = np.arctan((stroke[index+1][1]-stroke[index][1])/(stroke[index+1][0]-stroke[index][0]))
                else:
                    theta = 0
                dx = length_to_next_point*np.sin(theta)
                dy = length_to_next_point*np.cos(theta)
                repositioned.append([stroke[index][0]+dx, stroke[index][1]+dy])
                points_placed += 1
                length_to_next_point = len_separate
                stroke[index] = repositioned[-1]
            processed.append(repositioned)
        stroke_data[key] = processed
    return stroke_data

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
        raw_stroke_data = retrieve_stroke_data(X, directory, dataset_meta) 
        repositioned_stroke_data = reposition_xy_points(raw_stroke_data)
        break # Just for now to speed things up

extract_features("tmp/real-test.csv")
