import sys
import csv
import numpy as np

def usage(error = None):
    print "usage: python filter-class.py file class1 [class2 ... classN]"
    if error: print error

def main():
    args = sys.argv
    if len(args) < 3:
        usage()    
        return
    try:
        fname = args[1]
        with open(fname, 'r') as f:
            csvreader = csv.reader(f)
            arr = [row for row in csvreader]
            out = filter(lambda x: x[1].strip() in args[2:], arr)
        csvwriter = csv.writer(sys.stdout)
        for row in out:
            csvwriter.writerow(row)
    except IOError as e:
        usage(e.message)
    except Exception as e:
        usage()

if __name__ == "__main__":
    main()
