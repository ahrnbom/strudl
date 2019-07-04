""" A module for managing configurations for datasets and runs.
    Configurations exist so that various properties of an experiment
    can be defined once, instead of having to be repeated for every call.
    For example, the FPS and resolution of the videos is needed in many jobs.
    Note that there is special logic for reading/writing resolutions in a simple 
    string format.
"""

from pathlib import Path

from util import parse_resolution
from folder import datasets_path, runs_path, mkdir

class BaseConfig(object):
    contents = []
    data = None
    exists = False
    filepath = None
    
    def __init__(self, fp):
        """ Loads a dataset config object from a file """
        self.data = {}
        
        if (fp is not None) and fp.is_file():
            lines = fp.read_text().split('\n')
            
            for line in lines:
                if line.isspace():
                    continue
                
                if not line:
                    continue
                
                key, val = line.split(':')
                found = False
                for c in self.contents:
                    cname, ctype = c
                    if key == cname:
                        found = True
                        
                        val = self.parse_value(val, ctype)
                        
                        self.data[key] = val
                        
                assert(found)
            
            assert(len(self.data) == len(self.contents))
            self.exists = True            
    
    def parse_value(self, val, ctype):
        if ctype == 'res2':
            val = parse_resolution(val, 2)
            
            ok = True            
            for v in val:
                # Having resolutions not divisible by 16 causes issues with video encoding.
                if not (v%16 == 0):
                    ok = False
            if not ok:
                val = None
        
        elif ctype == 'res3':
            val = parse_resolution(val, 3)

            ok = True
            for v in val[0:-1]:
                if not (v%16 == 0):
                    ok = False            
            if not ok:
                val = None
        else:
            val = ctype(val)
            
        return val
    
    def save(self):
        assert(len(self.data) == len(self.contents))
        assert(not self.filepath is None)
        
        with self.filepath.open('w') as f:
            for key, val in self.data.items():
                line = "{}:{}\n".format(key, val)
                f.write(line)
    
    def get(self, key):
        return self.data[key]
        
    def get_data(self):
        """ Exports data such that it can be converted to JSON by connexion """
        export = {}
        for key, val in self.data.items():
            for c in self.contents:
                ckey, ctype = c
                if ckey == key:
                    if (type(ctype) == str) and (ctype[0:3] == 'res'):
                        val = str(val)
                    
            export[key] = val
        
        return export
    
    def set_data(self, imported):
        """ Imports data such that this config object can save its values """
        
        for c in self.contents:
            ckey, ctype = c
            if ckey in imported:
                val = imported[ckey]
                val = self.parse_value(val, ctype)
                if val is None:
                    return False
                self.data[ckey] = val
            else:
                return False
        
        self.exists = True
        return True
                    
class DatasetConfig(BaseConfig):
    contents = [('annotation_train_split', float), ('images_to_annotate', int),
                ('images_to_annotate_per_video', int), ('point_track_resolution', 'res2'),
                ('video_fps', int), ('video_resolution', 'res3')]
    
    def __init__(self, dataset):
        self.filepath = datasets_path / dataset / "config.txt"
        super().__init__(self.filepath)

class RunConfig(BaseConfig):
    contents = [('confidence_threshold', float), ('detection_batch_size', int),
                ('detection_training_batch_size', int), ('detector_resolution', 'res3')]
    
    def __init__(self, dataset=None, run=None):
        run_path = runs_path / "{}_{}".format(dataset,run)
        mkdir(run_path)
        self.filepath = run_path / "config.txt"
        super().__init__(self.filepath)
        
if __name__ == '__main__':
    # Simple test
    rc = RunConfig('test', 'testrun')
    print(rc.get('detector_resolution'))



            
        
    
        
