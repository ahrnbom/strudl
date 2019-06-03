""" A module for keeping track on folder-related things, like creating folders
    and keeping constants for commonly used paths
"""

import errno
from pathlib import Path
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
base_path = Path('/data')
datasets_path = str(base_path / 'datasets') + '/'
runs_path = str(base_path / 'runs') + '/'
jobs_path = str(base_path / 'jobs') + '/'
ssd_path = str(base_path / 'ssd') + '/'
