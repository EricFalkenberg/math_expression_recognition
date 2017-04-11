import sys
import csv
import glob, os





def main():
    NAME_LOCATION = 5
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python3 view-symbols.py <ground-truth.txt> [classes]")
        sys.exit(-1)
    print(args)
    drawClasses = args[1:]

    for c in drawClasses:
        with open(args[0]) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                file_name = row[0]
                for i in glob.glob("*.inkml"):
                    with open(i) as f2:
                        lines = f2.readlines()
                        print(lines[NAME_LOCATION])


main()