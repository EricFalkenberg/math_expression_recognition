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

def normalize_coords(stroke_data):
    new_data = {}
    num_points = len(stroke_data) 
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    print "Normalizing Coords"
    for key, strokes in stroke_data.items():
        tmp = []
        for stroke in strokes:
            tmp.extend(stroke)
        minY = min([i[1] for i in tmp])
        maxY = max([i[1] for i in tmp])
        
        new_strokes = []
        for stroke in strokes:
            if (maxY-minY != 0):
                new_strokes.append([[float((i[0]-minY))/(maxY-minY), float((i[1]-minY))/(maxY-minY)] for i in stroke])
            else:
                new_strokes.append([[0, 0] for i in stroke])
        new_data[key] = new_strokes 
        progress.update(curr)
        curr += 1
    return new_data

def smooth_xy_points(stroke_data):
    new_data = {}
    num_points = len(stroke_data) 
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    print "Smoothing Stroke Data"
    for key, value in stroke_data.items():
        smoothed = []
        for stroke in value:
            new_stroke = [(np.array(stroke[i-1]) + np.array(stroke[i]) + np.array(stroke[i+1])) / 3.0 \
                                                                            for i in range(1, len(stroke)-1)]
            new_stroke = [stroke[0]] + new_stroke + [stroke[-1]]
            smoothed.append(stroke)
        new_data[key] = smoothed
        progress.update(curr)
        curr += 1
    return new_data

def reposition_xy_points(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    ret_map = {}
    num_points = len(stroke_data) 
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    print "Resampling Points"
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
            points_to_place = np.round(percentage*60)
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
        progress.update(curr)
        curr += 1
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

def calc_ndtse(stroke_data):
    num_points = len(stroke_data) 
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    new_map = {}
    print 'Calculating NDTSE'
    for key, strokes in stroke_data.items():
        new_strokes = []
        lengths = [sum(distance(stroke[index], stroke[index+1]) for index in range(len(stroke)-1)) for stroke in strokes]
        for stroke, length in zip(strokes, lengths):
            st = []
            for p in stroke:
                db = np.abs(distance(stroke[0], p))
                de = np.abs(distance(p, stroke[-1]))
                if (length != 0):
                    st.append(1 - (de-db)/length)
                else:
                    st.append(0)
            new_strokes.append(st)
        new_map[key] = new_strokes
        progress.update(curr)
        curr += 1
    return new_map

def get_norm_y(stroke_data):
    num_points = len(stroke_data) 
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    new_map = {}
    print 'Retrieving Normalized Y Coordinate'
    for key, strokes in stroke_data.items():
        new_strokes = []
        for stroke in strokes:
            new_strokes.append([p[1] for p in stroke])
        new_map[key] = new_strokes
        progress.update(curr)
        curr += 1
    return new_map

def calc_vicinity_slope(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    num_points = len(stroke_data)
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    new_map = {}
    print 'Calculating Vicinity Slope'
    for key, strokes in stroke_data.items():
        new_strokes = []
        for stroke in strokes:
            p0  = 0
            p1  = 0
            st = []
            for idx in range(2, len(stroke)-2):
                last_points = stroke[idx-2]
                future_points = stroke[idx+2]
                diff_points = np.array(future_points)-np.array(last_points)
                dx = diff_points[0]
                dist = distance(last_points, future_points)
                if dist != 0:
                    angle = np.arcsin(dx / distance(last_points, future_points))
                else:
                    angle = 0
                st.append(angle+(np.pi/2))
            np1 = 0
            np0 = 0
            new_strokes.append([p0, p1] + st + [np1, np0])
        progress.update(curr)
        curr += 1
        new_map[key] = new_strokes
    return new_map

def calc_curvature(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    num_points = len(stroke_data)
    progress = progressbar.ProgressBar(max_value=num_points)
    curr = 0
    new_map = {}
    print 'Calculating Curvature' 
    for key, strokes in stroke_data.items():
        new_strokes = []
        for stroke in strokes:
            p0  = 0
            p1  = 0
            st = []
            for idx in range(2, len(stroke)-2):
                curr_points = stroke[idx]
                last_points = stroke[idx-2]
                future_points = stroke[idx+2]
                diff1 = np.array(curr_points)-np.array(last_points)
                diff2 = np.array(future_points)-np.array(curr_points)
                d1 = distance(last_points, curr_points)
                d2 = distance(curr_points, future_points)
                dx1 = diff1[0]
                dx2 = diff2[0]
                if (d1 != 0):
                    angle1 = np.arcsin(dx1 / d1)
                else:
                    angle1 = 0
                if (d2 != 0):
                    angle2 = np.arcsin(dx2 / d2)
                else:
                    angle2 = 0
                st.append(np.pi-angle1-angle2)
            np1 = 0
            np0 = 0
            new_strokes.append([p0, p1] + st + [np1, np0])
        progress.update(curr)
        curr += 1
        new_map[key] = new_strokes
    return new_map


def extract_features(fname, time_series=False):
    X, Y = load_data(fname)
    class_map = dict(zip(X, Y))
    dirs = ["%s/trainingSymbols/", "%s/trainingJunk/"]
    names   = []
    for directory in dirs:
        raw_stroke_data = retrieve_stroke_data(X, directory, dataset_meta) 
        norm_stroke_data = normalize_coords(raw_stroke_data)
        smoothed_stroke_data = smooth_xy_points(norm_stroke_data)
        repositioned_stroke_data = reposition_xy_points(smoothed_stroke_data)
        ndtse   = calc_ndtse(repositioned_stroke_data)
        norm_y  = get_norm_y(repositioned_stroke_data) 
        alpha   = calc_vicinity_slope(repositioned_stroke_data)
        beta    = calc_curvature(repositioned_stroke_data)
        flatten = lambda l: [item for sublist in l for item in sublist]
        dataset = [] 
        idx = 0
        for key, sample in ndtse.items():
            ndtse[key] = flatten(sample)
            norm_y[key] = flatten(norm_y[key])
            alpha[key] = flatten(alpha[key])
            beta[key] = flatten(beta[key])
            if ndtse[key] != []:
                if time_series:
                    for i in range(55):
                        if class_map[key] == '\\lambda':
                            print "LAMBDA"
                        names.append([key,class_map[key]]) 
                        
                        dataset.append([ndtse[key][i], norm_y[key][i], alpha[key][i], beta[key][i]])
                else:
                    names.append([key,class_map[key]]) 
                    dataset.append((ndtse[key][:55]+norm_y[key][:55]+alpha[key][:55]+beta[key][:55]))
        return names, dataset
