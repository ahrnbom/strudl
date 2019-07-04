""" A utility script for searching through all code for some phrase.
    Case-sensitive.
"""

import sys
from glob import glob

query = sys.argv[1]

files = glob('*.py')
files.sort()
for filepath in files:
    with open(filepath, 'r') as f:
        lines = [x.strip('\n') for x in f.readlines()]
        for line_number, line in enumerate(lines):
            if query in line:
                print("  {}, line {}".format(filepath, line_number+1))
                print("    {}".format(line.strip()))
                print("")
