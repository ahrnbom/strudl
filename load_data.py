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

    def custom(self, dataset, images_shape, train=True):
        # Note: images_shape should be the resolution of the training images, not the SSD resolution!!
        
        classnames = get_classnames(dataset)
        
        width = images_shape[0]
        height = images_shape[1]
        
        if train:
            trainval = 'train'
        else:
            trainval = 'val'
        
        all_gts = glob('{}{}/objects/{}/*/*.txt'.format(datasets_path, dataset, trainval))
        
        imfiles = []
        types = []
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for gt_txt in all_gts:
            imfile = gt_txt.replace('.txt', '.jpg')
            
            with open(gt_txt, 'r') as f:
                lines = [x.strip('\n') for x in f.readlines()]
                
            for line in lines:
                splot = line.split(' ')
                imfiles.append(imfile)        
                types.append(classnames[int(splot[0])-1])
                
                xc = width  * float(splot[1])
                yc = height * float(splot[2])
                bw = width  * float(splot[3])
                bh = height * float(splot[4])
                
                xmins.append(int(xc - bw/2))
                ymins.append(int(yc - bh/2))
                xmaxs.append(int(xc + bw/2))
                ymaxs.append(int(yc + bh/2))
        
        detections = pd.DataFrame()
        detections['image_file'] = imfiles
        detections['type'] = types
        detections['xmin'] = xmins
        detections['ymin'] = ymins
        detections['xmax'] = xmaxs
        detections['ymax'] = ymaxs
        
        return detections
                
