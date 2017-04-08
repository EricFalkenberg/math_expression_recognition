import os
import re
import csv
import sys
import xml.etree.ElementTree as ET

PAGE     = '<html>%s</html>'
HEADER   = '<h2 align="center">Classes: %s</h2>'
TABLE    = '<table style="width:80%%" align="center">%s</table>'
ROW      = '<tr>%s</tr>'
ITEM     = '<td>%s</td>'
SVG      = '<svg width="150" height="150">%s</svg>'
POLYLINE = '<polyline points="%s" style="fill:white;stroke:black;stroke-width:5;"/>'

def usage(error = None):
    print "usage: python view-symbols.py ground_truth.csv [limit]"
    if error: print error

def main():
    args = sys.argv
    if len(args) < 2:
        usage()
        return
    is_limit = len(args) == 3
    fname = args[1]
    classes   = set()
    class_map = {}
    with open(fname) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            class_map[row[0]] = row[1]
            if row[1] not in classes:
                classes.add(row[1])
    symbols = []
    count = 0 
    for filename in os.listdir("trainingSymbols"):
        if filename != "iso_GT.txt":
            tree = ET.parse("trainingSymbols/"+filename)
            root = tree.getroot() 
            annotations = root.findall("{http://www.w3.org/2003/InkML}annotation")       
            loc = annotations[1].text
            if loc not in class_map:
                continue
            count += 1
            trace = root.findall("{http://www.w3.org/2003/InkML}trace")
            lines = []
            for s in trace:
                stroke = [i.strip().split() for i in s.text.split(",")]
                stroke = [[float(i[0]), float(i[1])] for i in stroke]
                minX   = min(stroke, key=lambda x: x[0])[0]
                minY   = min(stroke, key=lambda x: x[1])[1]
                stroke = [[i[0]-minX, i[1]-minY] for i in stroke]
                maxX   = max(stroke, key=lambda x: x[0])[0]
                maxY   = max(stroke, key=lambda x: x[1])[1]
                try:
                    ratioX = 150.0 / maxX
                except:
                    ratioX = 0.0
                try: 
                    ratioY = 150.0 / maxY
                except:
                    ratioY = 0.0
                new_stroke = ' '.join([','.join([str(i[0]*ratioX), str(i[1]*ratioY)]) for i in stroke])
                lines.append(POLYLINE % new_stroke)
            symbol_info = (SVG % ''.join(lines), loc, class_map[loc])
            symbols.append(symbol_info)
            if is_limit:
                if count >= int(args[2]):
                    break

    rows = []
    curr_row = []
    for symbol, index in zip(symbols, range(len(symbols))):
        curr_row.append(ITEM % ' '.join(symbol))
        if index % 3 == 2:
            rows.append(ROW % ''.join(curr_row))
            curr_row = []
    table = TABLE % ''.join(rows)
    header = HEADER % ', '.join(classes)            
    print PAGE % ''.join([header, table]) 

main()
