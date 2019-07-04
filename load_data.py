""" A module for loading annotations to be fed into object detector training.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

from classnames import get_classnames
from folder import datasets_path


class LoadDetections():
    """ Loads the detections for different data sets.
        This class used to have different methods for loading 
        data from different sources. But since STRUDL enforces a specific
        folder layout, only one such method remains.
    """

    def custom(self, datasets, train=True):
        """ Parameters:
                datasets     - list of datasets to get images from. The first 
                               one in the list is the "main" one, whose classes 
                               will be used. It can also be a single string, if 
                               only a single dataset is to be used.
                train        - training or test set
        """
        
        if type(datasets) == str:
            datasets = [datasets]
        
        classnames = get_classnames(datasets[0])
        classnames_set = set(classnames)

        imfiles = []
        types = []
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        
        for dataset in datasets:
            this_classnames = get_classnames(dataset)
            
            if train:
                trainval = 'train'
            else:
                trainval = 'val'
            
            all_gts = list((datasets_path / dataset / "objects" / trainval).glob('*/*.txt'))
            
            if len(all_gts) == 0:
                raise(ValueError("Dataset '{}' doesn't have any grount truth files. Is it a correct dataset? Is there an extra space or something?".format(dataset)))
            
            for gt_txt in all_gts:
                imfile = gt_txt.with_suffix('.jpg')
                
                lines = gt_txt.read_text().split('\n')
                    
                for line in lines:
                    if not line:
                        continue
                
                    splot = line.split(' ')
                    imfiles.append(str(imfile))
                    
                    # Since we can import from different datasets, the class name needs to be
                    # checked, and marked as 'other' (the last class) if it doesn't exist
                    #types.append(classnames[int(splot[0])-1])
                    classname = this_classnames[int(splot[0])-1]
                    if classname in classnames_set:
                        types.append(classname)
                    else:
                        types.append(classnames[-1])
                    
                    xc = float(splot[1])
                    yc = float(splot[2])
                    bw = float(splot[3])
                    bh = float(splot[4])
                    
                    xmins.append((xc - bw/2))
                    ymins.append((yc - bh/2))
                    xmaxs.append((xc + bw/2))
                    ymaxs.append((yc + bh/2))
            
        detections = pd.DataFrame()
        detections['image_file'] = imfiles
        detections['type'] = types
        detections['xmin'] = xmins
        detections['ymin'] = ymins
        detections['xmax'] = xmaxs
        detections['ymax'] = ymaxs
            
        return detections

if __name__ == '__main__':
    ld = LoadDetections()
    d = ld.custom(['swemini','sweden2'])
    print(d)


