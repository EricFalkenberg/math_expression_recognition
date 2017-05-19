import math
from sklearn import preprocessing
from math import degrees, atan2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from config import file_handler_config as fconfig
from file_handler import read_training_data
from classification.features import smooth_xy_points, reposition_xy_points

def normalize_coords(strokes):
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
 
    minX = min([i[0] for i in tmp])
    maxX = max([i[0] for i in tmp])
    minY = min([i[1] for i in tmp])
    maxY = max([i[1] for i in tmp])

    scale_x = lambda x: (x-minX)*200./(maxY-minY)
    scale_y = lambda y: (y-minY)*200./(maxY-minY)
    new_strokes = []
    for stroke in strokes:
        if (maxY-minY != 0):
            new_strokes.append(
              [
                [scale_x(i[0]), scale_y(i[1])] \
                for i in stroke
              ]
            )
        else:
            new_strokes.append([[i[0]-minX, i[1]-minY] for i in stroke])

    return new_strokes

def create_image_from_points(strokes):
    stroke1 = strokes[0]
    bounding_box = calculate_bounding_box(stroke1)
    bbox_center  = (bounding_box[0]+bounding_box[1])/2., (bounding_box[2]+bounding_box[3])/2.

    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    upper_x = int(math.ceil(max([i[0] for i in tmp])))

    img = Image.new("RGB", (upper_x, 200), "white")
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        coord_pairs = [(i[0], i[1]) for i in stroke]
        for x,y in coord_pairs:
            draw.ellipse([(x,y),(x+3,y+3)], fill='black')
        #draw.line(coord_pairs, fill='black', width=5)
    draw.ellipse([(bbox_center[0],bbox_center[1]),(bbox_center[0]+5,bbox_center[1]+5)], fill='red')
    img.show()

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

def msscf(stroke1, stroke2):
    fix_angle = lambda x: 360+x if x < 0 else x
    bounding_box = calculate_bounding_box(stroke1)
    bbox_center  = (bounding_box[0]+bounding_box[1])/2., (bounding_box[2]+bounding_box[3])/2.

    strokes = [stroke1, stroke2]
    bins = [[0 for _ in range(5)] for _ in range(12)]

    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    upper_x = max([i[0] for i in tmp])
    tot = len(tmp)
    
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

def preprocess_strokes(traces):
    traces     = [i.data for i in traces]    
    smooth     = smooth_xy_points({'id':traces})
    reposition = reposition_xy_points(smooth)
    norm_y     = normalize_coords(reposition['id'])
    return norm_y

if __name__ == '__main__':
    dataset = read_training_data(fconfig['training_data_tiny'])
    idx = 0
    for i in dataset:
        f_handler = dataset[i]
        if not f_handler.is_malformed():
            express_traces = []
            for group in f_handler.groups:
                traces     = [i.data for i in group.traces]    
                smooth     = smooth_xy_points({'id':traces})
                reposition = reposition_xy_points(smooth)
                express_traces.extend(reposition['id'])
                norm_y     = normalize_coords(reposition['id'])
                if len(norm_y) > 1:
                    shape_context_features = msscf(norm_y[0], norm_y[1])
                    angles = [[i, i+30] for i in range(0, 360, 30)]
                    for c, a in zip(shape_context_features, angles):
                        print a
                        plt.bar(range(len(c)), c)
                        plt.show()
