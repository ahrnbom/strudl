""" Module for visualizing object detection annotations as a slidehsow-like video """

import imageio as io
import cv2

from load_data import LoadDetections
from visualize import class_colors, draw
from classnames import get_classnames
from apply_mask import Masker

def slideshow(dataset, outpath, fps=10, repeat=20):

    ld = LoadDetections()
    dets = ld.custom(dataset)

    imfiles = list(set(dets.image_file))
    if not imfiles:
        return False

    cc = class_colors()

    mask = Masker(dataset)

    classnames = get_classnames(dataset)

    with io.get_writer(outpath, fps=fps) as vid:
        for imfile in imfiles:
            d = dets[dets.image_file == imfile]
            
            # Add "class_name" and "class_index" columns which are missing
            d = d.rename(index=str, columns={"type":"class_name"})
            indices = [1+classnames.index(x) for x in d['class_name']]
            d['class_index'] = indices
            
            im = io.imread(imfile)
            im = mask.mask(im, alpha=0.5)
            
            width = float(im.shape[1])
            height = float(im.shape[0])
            frame = draw(im, d, cc, conf_thresh=-1.0, x_scale=width, y_scale=height)        
            
            for i in range(repeat):
                vid.append_data(frame)
    
    return True
                
if __name__ == '__main__':
    slideshow('rgb', 'slideshow.mp4')


