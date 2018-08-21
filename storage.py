""" A module for saving and loading python class instances to files. It was found that
    explicitly adding gzip compression compressed the files more than using pickle's
    own compression. In STRUDL, the .pklz ending is typically used for files compatible
    with this module.
"""

import gzip
import pickle

def save(data, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def load(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data

