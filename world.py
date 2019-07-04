""" A module for converting between pixel coordinates and world coordinates.
    Uses pdtv (https://bitbucket.org/hakanardo/pdtv/) code as a back-end,
    assumed to exist as a subfolder called 'pdtv'.
    Note that pdtv was written for Python 2, but STRUDL runs on Python 3. Therefore,
    a 2to3-converted version is included in STRUDL instead of just having pdtv as a git submodule.
"""

import pdtv.pdtv as pdtv
from folder import datasets_path

class Calibration(object):
    def __init__(self, dataset):
        params = ('dx','dy','Cx','Cy','Sx','f','k','Tx','Ty','Tz','r1','r2','r3','r4','r5','r6','r7','r8','r9')
        
        self.vals = {}
        
        lines = (datasets_path / dataset / "calib.tacal").read_text().split('\n')
        
        for line in lines:
            if not line:
                continue
        
            splot = line.split(' ')
            splot = [x for x in splot if x]
            
            param = splot[0].strip(':')
            assert(param in params)
            
            val = float(splot[1])
            self.vals[param] = val
        
        self.tsai = pdtv.TsaiCamera(**self.vals)
        
                
    def to_world(self, x, y, as_type=None, z=0.0):
        x, y, z = self.tsai.image_to_world((x,y), z)
        if not (as_type is None):
            x,y,z = map(as_type, (x,y,z))
        return (x,y,z)
        
    def to_pixels(self, x, y, z=0.0, as_type=None):
        x, y = self.tsai.world_to_image(x, y, z)
        if not (as_type is None):
            x,y = map(as_type, (x,y))
        return (x, y)

if __name__ == '__main__':
    c = Calibration('sweden2')
    
    # Sanity check: Take an image and a point, and draw points 1 m away in many directions
    
    import cv2
    im = cv2.imread('image.png')
    pos = (498,169)
    
    from math import cos,sin,pi
    from numpy import linspace
    
    pos_world = c.to_world(*pos)
    
    for angle in linspace(0,2*pi):
        x = pos_world[0] + 1*cos(angle)
        y = pos_world[1] + 1*sin(angle)
        
        pos_pixels = c.to_pixels(x,y)

        x,y = map(int, pos_pixels)
        
        cv2.circle(im, (x,y), 1, (255,255,255), -1)
    
    cv2.imwrite('world_sanity_check.png', im)

