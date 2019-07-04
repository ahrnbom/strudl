""" A module for checking if a point is within regions specified by mask images.
    While the images used are identical to the ones in apply_mask.py, the use-case
    is very different. This is not used much in STRUDL but could be useful if
    one would like to analyze tracks (like in the paper).
"""

import cv2

from folder import datasets_path
from util import clamp

class Check(object):
    def __init__(self, dataset, test, margin=0):
        filename = datasets_path / dataset / (test+'.png')
        if not filename.is_file():
            raise IOError("File not found: " + str(filename))
        self.mask = cv2.imread(str(filename), -1)
        self.mask = self.mask[:,:,3]
        self.w = self.mask.shape[1]-1
        self.h = self.mask.shape[0]-1
        
        if margin > 0:
            margin = int(round(margin))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(margin*2,margin*2))
            self.mask = cv2.dilate(self.mask, kernel)
            self.mask[0:margin, :] = 255
            self.mask[self.h-margin:, :] = 255
            self.mask[:, 0:margin] = 255
            self.mask[:, self.w-margin:] = 255
    
    def test(self, x, y):
        x = clamp(int(x), 0, self.w)
        y = clamp(int(y), 0, self.h)
        
        sampled = self.mask[y, x]
        
        if sampled > 127:
            return True
        
        return False
        
if __name__ == '__main__':
    c = Check('sweden2', 'vru_left')
    print('vru_left')
    print(' ', c.test(148,106))
    print(' ', c.test(343,162))
    
    c = Check('sweden2', 'mask', margin=10)
    print('mask')
    print(' ', c.test(473,175)) # in the non-masked region, should be False
    print(' ', c.test(313,43))  # clearly inside the masked region, should be True
    print(' ', c.test(329,108)) # close to the masked region, should be True
    print(' ', c.test(635,209)) # at the right edge of the screen, should be True


