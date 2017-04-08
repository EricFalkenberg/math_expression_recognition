import csv
import numpy as np
from config import dataset_meta

def load_data(fname):
    with open(fname) as f:
        csvreader = csv.reader(f)
        X = []
        Y = []
        for row in csvreader:
            X.append(row[:-1])
            Y.append(row[-1:])
    return X, Y

def retrieve_stroke_data(X, dataset_loc):
    pass

X, Y = load_data('tmp/real-train.csv')
retrieve_stroke_data(X, dataset_meta['location'])

