"""Classes for loading detections."""
import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from glob import glob

from classnames import get_classnames
from folder import datasets_path


class LoadDetections():
    """Loads the detections for different data sets."""

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
            
            all_gts = glob('{dsp}{ds}/objects/{t}/*/*.txt'.format(dsp=datasets_path, ds=dataset, t=trainval))
            
            if len(all_gts) == 0:
                raise(ValueError("Dataset '{}' doesn't have any grount truth files. Is it a correct dataset? Is there an extra space or something?".format(dataset)))
            
            for gt_txt in all_gts:
                imfile = gt_txt.replace('.txt', '.jpg')
                
                with open(gt_txt, 'r') as f:
                    lines = [x.strip('\n') for x in f.readlines()]
                    
                for line in lines:
                    splot = line.split(' ')
                    imfiles.append(imfile)        
                    
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


