import os
from bs4 import BeautifulSoup as Soup
import logging
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
        this.map_traces(trace_data)

    def map_traces(this, trace_data):
        this.traces = map(lambda x: trace_data[x], this.traces)
        
class f_handler:
    __slots__ = ('groups', 'traces')
    
    def __init__(this, groups, traces):
        this.groups = groups
        this.traces = traces

def read_inkml(fname):
    with open(fname) as f:
        soup = Soup(f)
        try:
            traces = { t.id: t for t in [trace(i) for i in soup.find_all('trace')] }
            groups = [group(i, traces) for i in soup.tracegroup.find_all('tracegroup')]
            return groups, traces
        except:
            return None, None

def read_directory(directory):
    out_map = {}
    files = os.listdir(directory)
    directory = directory+"/{}"
    for fn in files:
        fname = directory.format(fn)
        groups, traces = read_inkml(fname)
        out_map[fname] = f_handler(groups, traces)  
    return out_map

if __name__ == '__main__':
    dir_name = "/home/eric/Desktop/pattern_rec/TrainINKML/expressmatch"
    f_handlers = read_directory(dir_name)
    for i in f_handlers:
        print i

