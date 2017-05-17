from PIL import Image, ImageDraw
from config import file_handler_config as fconfig
from file_handler import read_training_data


def normalize_coords(strokes):
    tmp = []
    for stroke in strokes:
        tmp.extend(stroke)
 
    minX = min([i[0] for i in tmp])
    minY = min([i[1] for i in tmp])
    maxY = max([i[1] for i in tmp])
    
    new_strokes = []
    for stroke in strokes:
        if (maxY-minY != 0):
            new_strokes.append(
              [
                [200*float((i[0]-minX))/(maxY-minY), 200*float((i[1]-minY))/(maxY-minY)] \
                for i in stroke
              ]
            )
        else:
            new_strokes.append([[100, 100] for i in stroke])
    return new_strokes

def create_image_from_points(strokes):
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        coord_pairs = [(i[0], i[1]) for i in stroke]
        draw.line(coord_pairs, fill='black', width=5)
    img.show()

if __name__ == '__main__':
    dataset = read_training_data(fconfig['training_data_loc'])
    for i in dataset:
        f_handler = dataset[i]
        traces = f_handler.traces
        traces = [v.data for k,v in f_handler.traces.items()]    
        create_image_from_points(normalize_coords(traces))
