""" Module for applying masks to images. These masks are mostly transparent .png
    images with black regions covering non-interesting parts of the images, to 
    save annotation time mainly. This was previously quite slow and somewhat of
    a bottleneck so it was optimized a bit. Masks can be drawn transparently
    in visualization videos, or completely black when training the object detector.
"""

import cv2
import os
import numpy as np
from random import choice

from folder import datasets_path

class Masker(object):    
    def __init__(self, dataset):
        self.saved_mask = self.get_mask(dataset)
        
        # Cache alpha-adjusted masks because they can be slow to compute many times
        self.alpha_cache = {}
        self.alpha_cache_limit = 4

    def get_mask(self, dataset):
        mask_path = "{}{}/mask.png".format(datasets_path, dataset)
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, -1)           
        else:
            mask = None
            
        return mask
    
    def mask(self, im, alpha=1):
        mask = self.saved_mask
        
        if mask is None:
            return im
            
        alpha = float(alpha)
        n = int(round(1/alpha))
        
        if n in self.alpha_cache:
            mask_alpha3 = self.alpha_cache[n]
        else:    
            im_shape = im.shape
            mask_shape = mask.shape
            if not ((im_shape[0] == mask_shape[0]) and (im_shape[1] == mask_shape[1])):
                mask = cv2.resize(mask, (im_shape[1], im_shape[0]))
            
            mask_alpha = mask[:,:,3:]
            mask_alpha3 = np.zeros_like(im)
            for i in range(3):
                mask_alpha3[:,:,i] = mask_alpha[:,:,0]
            
            
            if not (n == 1):
                mask_alpha3 = mask_alpha3//n # This line is slow, but I don't know how it can be made faster
                
            # Cache this because it's slow to compute
            if len(self.alpha_cache) >= self.alpha_cache_limit:
                key = choice(tuple(self.alpha_cache.keys()))
                self.alpha_cache.pop(key)
                
            self.alpha_cache[n] = mask_alpha3
        
        # Necessary to prevent the image messing with the cache
        mask_alpha3 = mask_alpha3.copy()
        
        # Surprisingly, putmask is slower than doing it manually
        #mask_alpha3 = np.putmask(mask_alpha3, mask_alpha3>im, im)
        bigger = mask_alpha3 > im
        mask_alpha3[bigger] = im[bigger]
        
        im2 = im - mask_alpha3
        
        return im2
    
if __name__ == "__main__":
    im = cv2.imread('/data/datasets/sweden2/objects/train/4E40/12.jpg')
    masker = Masker("sweden2")
    masked = masker.mask(im)
    cv2.imwrite("applied_mask.png", masked)
