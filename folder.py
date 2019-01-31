""" A module for keeping track on folder-related things, like creating folders
    and keeping constants for commonly used paths
"""

import errno    
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass # Folder already exists, which is fine
        else:
            raise

# These are imported into many modules. The idea is that if one would want to
# change the paths, it would be sufficient to change the strings here.
base_path = 'data/'
datasets_path = 'data/datasets/'
runs_path = 'data/runs/'
jobs_path = 'data/jobs/'
