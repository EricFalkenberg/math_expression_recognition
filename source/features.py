import math
import matplotlib.pyplot as plt
import numpy
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
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    upper_x = int(math.ceil(max([i[0] for i in tmp])))

    img = Image.new("RGB", (upper_x, 200), "white")
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        coord_pairs = [(i[0], i[1]) for i in stroke]
        draw.line(coord_pairs, fill='black', width=5)
    img.show()

def get_angle_bin(angle):
    idx = 0
    while angle > 0:
        angle -= 30
        idx += 1
    return idx

def get_distance_bin(d):
    dists = [1./16, 1./8, 1./4, 1./2, 1.]
    idx = 0
    while d > dists[idx] and idx < len(dists)-1:
        idx += 1
    return idx

def msscf(strokes):
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
    upper_x = max([i[0] for i in tmp])
    tot = len(tmp)

    bins = [[0 for _ in range(5)] for _ in range(12)]
    ns = [[[i[0]/200, i[1]/200.] for i in stroke] for stroke in strokes]

    base_x = upper_x / 2.
    base_y = 0.5
 
    for stroke in strokes:
        for x,y in stroke:
            dist  = distance(base_x, base_y, x, y)
            d_bin = get_distance_bin(dist) 
            dx = x-base_x
            dy = y-base_y
            angle = math.degrees(math.atan(float(dy)/dx))
            a_bin = get_angle_bin(angle)
            bins[a_bin][d_bin] += 1
    return [[float(i)/tot for i in x] for x in bins]
     

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
    d_between_bc = distance(b_center1[0], b_center2[0], b_center1[1], b_center2[1]) 
    avg_center1  = sum([i[0] for i in stroke1])/float(len(stroke1)), \
                   sum([i[1] for i in stroke1])/float(len(stroke1))
    avg_center2  = sum([i[0] for i in stroke2])/float(len(stroke2)), \
                   sum([i[1] for i in stroke2])/float(len(stroke2))
    d_between_ac = distance(avg_center1[0], avg_center2[0], avg_center1[1], avg_center2[1])
    print d_between_bc, d_between_ac


if __name__ == '__main__':
    dataset = read_training_data(fconfig['training_data_tiny'])
    idx = 0
    for i in dataset:
        f_handler = dataset[i]
        if not f_handler.is_malformed():
            express_traces = []
            print i
            for group in f_handler.groups:
                #if len(group.traces) > 1:
                traces     = [i.data for i in group.traces]    
                smooth     = smooth_xy_points({'id':traces})
                reposition = reposition_xy_points(smooth)
                #stroke_symbol_pair_features(reposition['id'][0], reposition['id'][1])
                express_traces.extend(reposition['id'])
            norm_y     = normalize_coords(express_traces)
            create_image_from_points(norm_y)
            shape_context_features = msscf(norm_y)
            for c in shape_context_features:
                plt.bar(range(len(c)), c)
                plt.show()
