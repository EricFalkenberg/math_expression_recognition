"""
File: filter-class.py - Assignment 1
Author: Ryan Tompkins
Class: Pattern Recognition
Prof: Professor Zanibbi

Question 1 for Assignment 3
"""

import sys
import csv
import random

# filenames that this script writes to
filenames = ["junk-train.csv", "junk-test.csv",
             "real-train.csv", "real-test.csv"]

"""
Run this script with any two CSV files as input
"""
def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("Usage: python3 split-class.py <real_data_file> <junk_data> <t>")
        sys.exit(-1)

    realDataFile = args[0]
    junkDataFile = args[1]
    randomT      = float(args[2]) / 100

    realData      = []   # this will be used as the training data later
    junkData      = []

    with open(realDataFile) as f:
        f_csv = csv.reader(f)
        numSamples = 0
        for row in f_csv:
            numSamples = numSamples + 1
            realData.append([row[0], row[1]])

    realTestData = random.sample(realData, round(randomT * numSamples))

    with open(junkDataFile) as f:
        f_csv = csv.reader(f)
        numSamples = 0
        for row in f_csv:
            numSamples = numSamples + 1
            junkData.append([row[0], row[1]])

    junkTestData = random.sample(junkData, round(randomT * numSamples))

    # remove the test samples from the data sets to use them as the training data
    for i in range(0, len(realTestData)):
        realData.remove(realTestData[i])
    for i in range(0, len(junkTestData)):
        junkData.remove(junkTestData[i])

    dataList = [junkData, junkTestData, realData, realTestData]
    filenames = ["junk-train.csv", "junk-test.csv",
                 "real-train.csv", "real-test.csv"]

    for i in range(len(dataList)):
        with open(filenames[i], "w") as f:
            f_csv = csv.writer(f)
            f_csv.writerows(dataList[i])

main()