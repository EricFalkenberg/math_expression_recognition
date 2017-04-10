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

def viz(strokes):
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    minX   = min([i[0] for i in tmp])
    minY   = min([i[1] for i in tmp])
    maxX   = max([i[0] for i in tmp])
    maxY   = max([i[1] for i in tmp])

    new_strokes = []
    for stroke in strokes:
        if (maxX-minX != 0) and (maxY-minY != 0):
            new_strokes.append([[float((i[0]-minX))/(maxX-minX), float((i[1]-minY))/(maxY-minY)] for i in stroke])
    ret = ''
    for stroke in new_strokes:
        ret += ' '.join([','.join([str(i[0]*300), str(i[1]*300)]) for i in stroke])
        ret += '\n'*2
    return ret

def smooth_xy_points(stroke_data):
    new_data = {}
    for key, value in stroke_data.items():
        smoothed = []
        for stroke in value:
            new_stroke = [(np.array(stroke[i-1]) + np.array(stroke[i]) + np.array(stroke[i+1])) / 3.0 \
                                                                            for i in range(1, len(stroke)-1)]
            new_stroke = [stroke[0]] + new_stroke + [stroke[-1]]
            smoothed.append(stroke)
        new_data[key] = smoothed
    return new_data

def reposition_xy_points(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    ret_map = {}
    for key, value in stroke_data.items():
        # Populate list representing the length of each stroke
        segments_length = [sum([distance(s[x], s[x+1]) for x in range(len(s)-1)]) for s in value]
        # Calculate total length of symbol strokes combined
        stroke_length = sum(segments_length)
        # Estimate the percentage of each stroke length in relation to total symbol length
        percentages = [float(i)/stroke_length if stroke_length != 0 else 0 for i in segments_length]
        processed = []
        # Iterate through all strokes
        for stroke, percentage, length in zip(value, percentages, segments_length):
            # Determine the number of points to place on the current stroke
            st = stroke
            points_to_place = np.round(percentage*50)
            if points_to_place == 0:
                continue
            points_placed = 1
            repositioned = [st[0]]
            # Distance between each new point
            len_separate = float(length) / points_to_place
            
            length_to_next_point = len_separate
            index = 0
            while points_placed < points_to_place:
                # While our next point isnt between index and index+1, increment
                while length_to_next_point > distance(st[index], st[index+1]):
                    length_to_next_point -= distance(st[index], st[index+1])
                    index += 1
                od = distance(st[index], st[index+1])
                t  = length_to_next_point / distance(st[index], st[index+1])
                new_x = (1 - t)*st[index][0] + t*st[index+1][0]
                new_y = (1 - t)*st[index][1] + t*st[index+1][1] 
                # Add new point to repositioned array
                repositioned.append([new_x, new_y])
                points_placed += 1
                # Reset length to next point and set current index to point placed
                length_to_next_point = len_separate
                st[index] = repositioned[-1]
            processed.append(repositioned)
        ret_map[key] = processed
    return ret_map

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
        smoothed_stroke_data = smooth_xy_points(raw_stroke_data)
        repositioned_stroke_data = reposition_xy_points(smoothed_stroke_data)
        break # Just for now to speed things up

extract_features("tmp/tmp.csv")
