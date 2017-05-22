from config import dataset_meta
import math
from math import degrees, atan2
import csv
import os
import progressbar
import numpy as np
import xml.etree.ElementTree as ET

############################
def get_angle_bin(angle):
    if 0 <= angle < 30 or angle == 360:
        return 0
    if 30 <= angle < 60:
        return 1
    if 60 <= angle < 90:
        return 2
    if 90 <= angle < 120:
        return 3
    if 120 <= angle < 150:
        return 4
    if 150 <= angle < 180:
        return 5
    if 180 <= angle < 210:
        return 6
    if 210 <= angle < 240:
        return 7
    if 240 <= angle < 270:
        return 8
    if 270 <= angle < 300:
        return 9
    if 300 <= angle < 330:
        return 10
    if 330 <= angle < 360:
        return 11

def get_distance_bin(d):
    dists = [200./16, 200./8, 200./4, 200./2, 200.]
    if d < dists[0]:
        return 0
    if d < dists[1]:
        return 1
    if d < dists[2]:
        return 2
    if d < dists[3]:
        return 3
    if d < dists[4]:
        return 4
    else:
        return 4

def msscf(strokes):
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    tot = len(tmp)

    fix_angle = lambda x: 360+x if x < 0 else x
    bounding_box = calculate_bounding_box(tmp)
    bbox_center  = (bounding_box[0]+bounding_box[1])/2., (bounding_box[2]+bounding_box[3])/2.

    bins = [[0 for _ in range(5)] for _ in range(12)]

    
    base_x, base_y = bbox_center
 
    for stroke in strokes:
        for x,y in stroke:
            dist  = distance(base_x, x, base_y, y)
            d_bin = get_distance_bin(dist) 
            dx = x-base_x
            dy = y-base_y
            dot = base_x*x + base_y*y
            det = base_x*y - base_y*x
            angle = fix_angle(degrees(atan2(float(dy), float(dx))))
            a_bin = get_angle_bin(angle)
            bins[a_bin][d_bin] += 1
    output = []
    for x in bins:
        output.extend([float(i)/tot for i in x])
    return output 

def calculate_angle(x1,x2,y1,y2):
    fix_angle = lambda x: 360+x if x < 0 else x
    dx = x2-x1
    dy = y2-y1
    return fix_angle(degrees(atan2(float(dy), float(dx))))
     

def distance(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def calculate_bounding_box(stroke):
    minX = min([i[0] for i in stroke])
    maxX = max([i[0] for i in stroke])
    minY = min([i[1] for i in stroke])
    maxY = max([i[1] for i in stroke])
    return minX, maxX, minY, maxY

def stroke_symbol_pair_features(stroke1, stroke2):
    bounding1    = calculate_bounding_box(stroke1)
    bounding2    = calculate_bounding_box(stroke2)
    b_center1    = (bounding1[0]+bounding1[1])/2.0, (bounding1[2]+bounding1[3])/2.0
    b_center2    = (bounding2[0]+bounding2[1])/2.0, (bounding2[2]+bounding2[3])/2.0
    avg_center1  = sum([i[0] for i in stroke1])/float(len(stroke1)), \
                   sum([i[1] for i in stroke1])/float(len(stroke1))
    avg_center2  = sum([i[0] for i in stroke2])/float(len(stroke2)), \
                   sum([i[1] for i in stroke2])/float(len(stroke2))
    ## Distance between bounding box centers
    d_between_bc = distance(b_center1[0], b_center2[0], b_center1[1], b_center2[1]) 
    ## Distance between averaged centers
    d_between_ac = distance(avg_center1[0], avg_center2[0], avg_center1[1], avg_center2[1])
    ## Horizontal offset between end of first and beginning of second
    h_offset  = stroke2[0][0] - stroke1[-1][0] 
    ## Vertical offset between bounding box centers
    v_dist_bb = b_center2[1] - b_center1[1] 
    ## Get writing slope
    writing_slope = calculate_angle(stroke1[-1][0],stroke1[-1][1],stroke2[0][0],stroke2[0][1])
    ## Get maximum point pair distance
    max_pp_dist  = 0
    for x1,y1 in stroke1:
        for x2,y2 in stroke2:
            d = distance(x1,x2,y1,y2)
            if d > max_pp_dist:
                max_pp_dist = d
    g_features = [d_between_bc, d_between_ac, max_pp_dist, h_offset, v_dist_bb, writing_slope]
    return preprocessing.scale(g_features)


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
    curr = 0
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
    return new_data

def smooth_xy_points(stroke_data):
    new_data = {}
    num_points = len(stroke_data) 
    curr = 0
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
    num_points = len(stroke_data) 
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
                if len(st) > 0:
                     repositioned = [st[0] for _ in range(100)]
                     processed.append(repositioned)
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
    for filename in os.listdir(directory % config['location']):
        if filename not in config['exclude']:
            tree = ET.parse((directory+"%s") % (config['location'], filename))
            root = tree.getroot() 
            annotations = root.findall(config['xml_name_tag'])       
            loc = annotations[1].text
            trace = root.findall(config['xml_trace_tag'])
            tmp = map(extract_xy_data, (stroke.text for stroke in trace))
            if not (any([len(i)<2 for i in tmp]) or len(tmp) < 1):
                 trace_map[loc] = map(extract_xy_data, (stroke.text for stroke in trace))
    return {i:trace_map[i] for i in X if i in trace_map}

def calc_ndtse(stroke_data):
    num_points = len(stroke_data) 
    curr = 0
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    new_map = {}
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
    return new_map

def get_norm_y(stroke_data):
    num_points = len(stroke_data) 
    new_map = {}
    for key, strokes in stroke_data.items():
        new_strokes = []
        for stroke in strokes:
            new_strokes.append([p[1] for p in stroke])
        new_map[key] = new_strokes
    return new_map

def calc_vicinity_slope(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    num_points = len(stroke_data)
    new_map = {}
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
        new_map[key] = new_strokes
    return new_map

def calc_curvature(stroke_data):
    distance = lambda p1, p2: np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    num_points = len(stroke_data)
    new_map = {}
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
        new_map[key] = new_strokes
    return new_map

def extract_features_from_sample(strokes):
    stroke_map = { 'id' : strokes }
    smoothed_stroke_data = smooth_xy_points(stroke_map)
    repositioned_stroke_data = reposition_xy_points(smoothed_stroke_data)
    norm_stroke_data = normalize_coords(repositioned_stroke_data)
    shape_context      = shape_cont(norm_stroke_data)
    ndtse   = calc_ndtse(norm_stroke_data)
    norm_y  = get_norm_y(norm_stroke_data) 
    alpha   = calc_vicinity_slope(norm_stroke_data)
    beta    = calc_curvature(norm_stroke_data)
    flatten = lambda l: [item for sublist in l for item in sublist]
    idx = 0
    dataset = [] 
    for key, sample in ndtse.items():
        ndtse[key] = flatten(sample)
        norm_y[key] = flatten(norm_y[key])
        alpha[key] = flatten(alpha[key])
        beta[key] = flatten(beta[key])
        if ndtse[key] != []:
            dataset.append((shape_context[key]+ndtse[key][:45]+norm_y[key][:45]+alpha[key][:45]+beta[key][:45]))
    return dataset
    
def shape_cont(stroke_map):
    new_map = {}
    for key, strokes in stroke_map.items():
        new_map[key] = msscf(strokes)
    return new_map

def extract_features(fname, time_series=False):
    X, Y = load_data(fname)
    class_map = dict(zip(X, Y))
    dirs = ["%s/trainingSymbols/", "%s/trainingJunk/"]
    names   = []
    dataset = [] 
    for directory in dirs:
        raw_stroke_data = retrieve_stroke_data(X, directory, dataset_meta) 
        smoothed_stroke_data = smooth_xy_points(raw_stroke_data)
        repositioned_stroke_data = reposition_xy_points(smoothed_stroke_data)
        norm_stroke_data = normalize_coords(repositioned_stroke_data)
        ndtse   = calc_ndtse(norm_stroke_data)
        norm_y  = get_norm_y(norm_stroke_data) 
        alpha   = calc_vicinity_slope(norm_stroke_data)
        beta    = calc_curvature(norm_stroke_data)
        shape_context      = shape_cont(norm_stroke_data)
        flatten = lambda l: [item for sublist in l for item in sublist]
        idx = 0
        for key, sample in ndtse.items():
            ndtse[key] = flatten(sample)
            norm_y[key] = flatten(norm_y[key])
            alpha[key] = flatten(alpha[key])
            beta[key] = flatten(beta[key])
            if ndtse[key] != []:
                if time_series:
                    for i in range(55):
                        names.append([key,class_map[key]]) 
                        dataset.append([ndtse[key][i], norm_y[key][i], alpha[key][i], beta[key][i]])
                else:
                    names.append([key,class_map[key]]) 
                    dataset.append((shape_context[key]+ndtse[key][:45]+norm_y[key][:45]+alpha[key][:45]+beta[key][:45]))
        return names, dataset
