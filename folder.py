""" A module for keeping track on folder-related things, like creating folders
    and keeping constants for commonly used paths
"""

import errno    
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

base_path = '/data/'
datasets_path = '/data/datasets/'
runs_path = '/data/runs/'
jobs_path = '/data/jobs/'
