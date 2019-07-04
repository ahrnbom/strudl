""" A module for keeping track on folder-related things, like creating folders
    and keeping constants for commonly used paths
"""

import errno
from pathlib import Path
import os

def mkdir(path):
    if type(path) == str:
        try:
            os.makedirs(path)
        except OSError as exc:  
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass # Folder already exists, which is fine
            else:
                raise
    elif isinstance(path, Path): # This works despite the path typically being a PosixPath
        path.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError("Provided path is of unsupported type ({}), try a string or a Path object.".format(type(path)))

# These are imported into many modules. The idea is that if one would want to
# change the paths, it would be sufficient to change the paths here.
base_path = Path('/data')
datasets_path = base_path / 'datasets'
runs_path = base_path / 'runs'
jobs_path = base_path / 'jobs'
ssd_path = base_path / 'ssd'
