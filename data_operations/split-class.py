import sys
import csv
import numpy as np

def usage(error = None):
    print "usage: python split-class.py t iso_GT junk_GT"
    if error: print "ERROR: %s" % error

def main():
    args = sys.argv
    if len(args) != 4:
        usage()    
        return
    try:
        t = int(args[1])
        if not (0 < t < 100):
            usage("ERROR: t value must be between 0 and 100")
        ## Read real data from isoGT
        isoGT_name = args[2]
        with open(isoGT_name) as f:
            csvreader = csv.reader(f)
            arr = np.array([row for row in csvreader])
            train = arr[np.random.randint(arr.shape[0], size=np.round(len(arr)*(100-t)/100)), :]
            test  = arr[np.random.randint(arr.shape[0], size=np.round(len(arr)*t/100)), :] 
        ## Write training data to real-train.csv
        with open("real-train.csv", "w") as f:
            csvwriter = csv.writer(f)
            for row in train:
                csvwriter.writerow(row)
        ## Write testing data to real-test.csv
        with open("real-test.csv", "w") as f:
            csvwriter = csv.writer(f)
            for row in test:
                csvwriter.writerow(row)
        ## Read junk data from junkGT 
        junkGT_name = args[3]
        with open(junkGT_name) as f:
            csvreader = csv.reader(f)
            arr = np.array([row for row in csvreader])
            train = arr[np.random.randint(arr.shape[0], size=np.round(len(arr)*(100-t)/100)), :]
            test  = arr[np.random.randint(arr.shape[0], size=np.round(len(arr)*t/100)), :] 
        ## Write training data to real-train.csv
        with open("junk-train.csv", "w") as f:
            csvwriter = csv.writer(f)
            for row in train:
                csvwriter.writerow(row)
        ## Write testing data to real-test.csv
        with open("junk-test.csv", "w") as f:
            csvwriter = csv.writer(f)
            for row in test:
                csvwriter.writerow(row)
        
    except ValueError as e:
        usage(e.message)
    except IOError as e:
        usage(e.message)

if __name__ == "__main__":
    main()
