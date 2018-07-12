import cv2 
import pandas as pd
import os
import numpy as np
import sys
from glob import glob
import click

from apply_mask import Masker
from util import parse_resolution, pandas_loop

def class_colors(num_classes=10):
    """Generates num_classes many distinct colors"""
    class_colors = []
    for i in range(0, num_classes):
        # This can probably be written in a more elegant manner
        hue = 255*i/(num_classes+2)
        col = np.zeros((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 # Saturation
        col[0][0][2] = 255 # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col) 
    return class_colors

def draw_arrow(to_draw, cx, cy, xx, yy, cname, cindex, class_colors, conf=None, extratext=""):
    """ Draws an arrow on top of an image. The image is then returned. Assumes coordinates are integers in pixel coordinates.
        Arguments: 
        to_draw             -- image to draw on 
        cx                  -- center x coordinate
        cy                  -- center y coordinate
        xx                  -- outer x coordinate (that arrow should point at)
        yy                  -- outer y coordinate (that arrow should point at)
        cname               -- class name
        cindex              -- index of class
        class_colors        -- list of colors for each class
        conf                -- confidence threshold (set to None to keep every detection)
    """
    
    col = class_colors[cindex]
    
    cv2.arrowedLine(to_draw, (cx,cy), (xx,yy), col, 3, cv2.LINE_AA, 0, 0.5)
    
    text = cname
    if conf is not None:
        text += " " + ('%.2f' % conf)
    
    if extratext:
        text += " " + extratext
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    
    text_top = (cx - text_size[0][0]//2, cy - text_size[0][1]//2-2)
    text_bot = (cx + text_size[0][0]//2, cy + text_size[0][1]//2+2)
    text_pos = (cx - text_size[0][0]//2, cy + text_size[0][1]//2)
    
    cv2.rectangle(to_draw, text_top, text_bot, col, -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)
    
    return to_draw

def draw_box(to_draw, xmin, xmax, ymin, ymax, cname, cindex, class_colors, conf=None, extratext=""):
    """ Draws a box on top of an image. The image is then returned.
        Arguments: 
        to_draw                -- image to draw on
        xmin, xmax, ymin, ymax -- coordinates of the box
        cname                  -- class name, to be written on the box
        cindex                 -- class index, used for determining the color
        class_colors           -- a list of mutliple colors
        conf                   -- confidence of box, to be written
        extratext              -- some other additional text to be written
    """

    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), class_colors[cindex], 2)
    text = cname
    if conf is not None:
        text += " " + ('%.2f' % conf)
    
    text += " " + extratext
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    text_top = (xmin, ymin-15)
    text_bot = (xmin + text_size[0][0]+10, ymin-10+text_size[0][1])
    text_pos = (xmin + 5, ymin-2)

    cv2.rectangle(to_draw, text_top, text_bot, class_colors[cindex], -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)
    
    return to_draw

def draw(to_draw, df, class_colors, conf_thresh=0.7, x_scale=1.0, y_scale=1.0, coords='pixels', calib=None): 
    """ Draws boxes from a data frame to an image, which is then returned.
        Arguments:
        to_draw          -- an image to draw on
        df               -- data frame with object detections
        class_colors     -- list of colors
        conf_thresh      -- threshold of confidence, detection below this are not included. If negative, confidences are not used at all.
        x_scale, y_scale -- scales the coordinates from the data frame in case the image is of another resolution
        coords           -- 'pixels' for normal pixel coordinates, 'world' for special treatment for world coordinates visualization including movement direction
        calib            -- if in world coordinates, a Calibration object (from world.py module)
    """

    noconf = False
    if conf_thresh < 0:
        noconf = True
 
    if noconf or (conf_thresh == 0.0): # checking for 0.0 here isn't necessary but skips the somewhat slow pandas operation
        df2 = df
    else:
        df2 = df.loc[df['confidence'] > conf_thresh]
    
    if coords == 'pixels':        
        for row in pandas_loop(df2):
            xmin = int(row['xmin']*x_scale)
            xmax = int(row['xmax']*x_scale)
            ymin = int(row['ymin']*y_scale)
            ymax = int(row['ymax']*y_scale)
            cname = row['class_name']
            cindex = row['class_index']
            conf = None
            if not noconf:
                conf = row['confidence']
            
            to_draw = draw_box(to_draw, xmin, xmax, ymin, ymax, cname, cindex, class_colors, conf=conf)
    elif coords == 'world':
        for row in pandas_loop(df2):
            wx = row['world_x']
            wy = row['world_y']
            
            wdx = row['world_dx']
            wdy = row['world_dy']
            
            cname = row['class_name']
            cindex = row['class_index']
            
            cx, cy = calib.to_pixels(wx, wy, as_type=int)
            xx, yy = calib.to_pixels(wx+wdx, wy+wdy, as_type=int)
            
            conf = None
            if not noconf:
                conf = row['confidence']
            
            to_draw = draw_arrow(to_draw, cx, cy, xx, yy, cname, cindex, class_colors, conf=conf)
    else:
        raise(ValueError("Incorrect coords {}".format(coords)))
        
    return to_draw
    
def parse(basepath, dataset, resolution):
    """ Parses a dataset for data frames CSV files and draws the detections.
        This is used for showing object detector results on the validation set used 
        during training. 
        Arguments:
        basepath -- path to folder with CSV files
        dataset  -- name of the dataset used, used for finding the correct mask
    """
    colors = class_colors()
    masker = Masker(dataset)

    csvpath = basepath + 'detections_0.csv'
    res = pd.read_csv(csvpath)
    
    outpath = basepath + 'visualize/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        old_files = glob(outpath + '*')
        for old in old_files:
            os.remove(old)
    
    files = res['filename'].unique()
    for i, filename in enumerate(files):
        df = res.loc[res['filename'] == filename]
        impath = df['filename'].iloc[0]
        im = cv2.imread(impath)
        im = cv2.resize(im, (resolution[0], resolution[1]))
        im = masker.mask(im)
        im = draw(im, df, colors)
        outfilepath = "{}{}".format(outpath, '{}_{}'.format(1+i, filename.split('/')[-1]))
        cv2.imwrite(outfilepath, im)
        print(outfilepath)
    
@click.command()
@click.option("--dataset", default="sweden2", help="Name of the dataset to use")
@click.option("--run", default="default", help="Name of the training run to use")
@click.option("--res", default="(640,480,3)", help="Resolution that SSD is trained for, as a string like '(width,height,channels)'")
def main(dataset, run, res):
    res = parse_resolution(res)
    parse('/data/dl/{}_{}/results/'.format(dataset, run), dataset, res)
    
if __name__ == "__main__":
    main()

    


