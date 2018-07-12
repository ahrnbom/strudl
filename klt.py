""" A module for creating KLT point tracks in videos. Can be run directly on videos
    with minor or no pre-processing, and provides information about how objects move
    in the video.
"""

import cv2
import numpy as np
import imageio as io
from itertools import count
from time import monotonic as time
import math

import sys
from glob import glob
import pickle
import click

from folder import mkdir, datasets_path
from storage import save
from util import parse_resolution, print_flush

def get_colors(n=10):
    colors = []
    for i in range(0, n):
        # This can probably be written in a more elegant manner
        hue = 255*i/(n+1)
        col = np.zeros((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 200 # Saturation
        col[0][0][2] = 255 # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        colors.append(col) 
    return colors

class Track(list):
    """ A KLT track. Each track needs an ID number """
    id_num = -1

def kltfull(video_file, imsize, mask, out_file=None):
    """ Performs KLT point tracking on a video.
    
    Arguments:
    video_file -- path to a source video file
    imsize     -- size which frames will be resized to
    mask       -- a mask which can be applied to only look at parts of the images
    out_file   -- if set to a path to an output video path, then a video showing
                    the tracked points is created. Can be None, in which case no
                    video is made
    """
    render_vid = True
    if out_file is None:
        render_vid = False

    track_len = 10
    detect_interval = 10
    tracks = []
    frame_idx = 0
    
    if render_vid: 
        n_colors = 128
        colors = get_colors(n_colors)
            
    id_generator = count()
    
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 1,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 5000,
                           qualityLevel = 0.01,
                           minDistance = 30,
                           blockSize = 7 )

    lost_tracks = []
    start_time = time()
    
    if render_vid:
        avi = io.get_writer(out_file, fps=10)
        
    with io.get_reader(video_file) as invid:
        vidlength = len(invid)
        for systime, frame in enumerate(invid):
            
            if systime % 400 == 0:
                print_flush("{} % done, elapsed time: {} s".format(round(100*systime/vidlength), round(time() - start_time)))
            
            frame = cv2.resize(frame, imsize)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()


            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1][1:3] for tr in tracks]).reshape(-1, 1, 2)
                p1, st, err1 = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag, e in zip(tracks, p1.reshape(-1, 2), good, err1.flat):
                    if not good_flag:
                        lost_tracks.append(tr)
                        continue
                    tr.append((systime, x, y))
                    #if len(tr) > track_len:
                    #    del tr[0]
                    new_tracks.append(tr)
                    if render_vid: 
                        cv2.circle(vis, (x, y), 2, colors[tr.id_num % n_colors], -1)
                tracks = new_tracks
                if render_vid:
                    for i_col, col in enumerate(colors):
                        cv2.polylines(vis, [np.int32([(x,y) for f,x,y in tr[-20:]]) for tr in tracks if (tr.id_num % n_colors) == i_col],
                                      False, col)

            if frame_idx % detect_interval == 0:
                mask2 = mask.copy()
                for x, y in [np.int32(tr[-1][1:3]) for tr in tracks]:
                    cv2.circle(mask2, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask2, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        nt = Track([(systime, x, y)])
                        nt.id_num = next(id_generator)
                        tracks.append(nt)

            frame_idx += 1
            prev_gray = frame_gray
                       
            if render_vid:     
                avi.append_data(vis)
    lost_tracks.extend(tracks)
    
    if render_vid:
        avi.close()
        
    return lost_tracks
    
def klt_save(vidpath, datpath, imsize, mask, outvidpath=None):
    """ Computes and saves KLT point tracks
        
        Arguments:
        vidpath    -- path to input video
        datpath    -- path to store the tracks (use .pklz extension)
        imsize     -- size to resize frames to 
        mask       -- mask to apply if only parts of the image are of interest
        outvidpath -- path to output video, can be None
    """
    tracks = kltfull(vidpath, imsize, mask, outvidpath)
    
    compact_tracks = []
    for tr in tracks:
        compact = []
        for det in tr:
            det2 = [int(round(x)) for x in det]
            compact.append(det2)

        compact_tracks.append(compact)
            
    save(compact_tracks, datpath)
    
#    with open(csvpath, 'w') as f:
#        f.write("id,frame_number,x,y\n")
#        for i, tr in enumerate(tracks):
#            for t, x, y, _ in tr:
#                f.write("{},{},{:.2f},{:.2f}\n".format(i, t, x, y))

@click.command()
@click.option("--cmd", default="findvids", help="Which command to run, either 'findvids' to search for videos, 'continue' to keep running a cancelled run or 'test' to run a test on some hardcoded video")
@click.option("--dataset", default="sweden2", help="Which dataset to run on")
@click.option("--imsize", default="(320,240)", help="Image size to run KLT on (smaller is much faster), as a string on this format: '(320,240)' where 320 is the width and 240 is the height")
@click.option("--nice", default=10, help="Niceness of this program (to allow other processes to use the hardware resources)")
def main(cmd, dataset, imsize, nice):
    import os
    os.nice(nice)
    
    imsize = parse_resolution(imsize)
    
    maskpath = "{}{}/mask.png".format(datasets_path, dataset)
    mask = 255 - cv2.resize(cv2.imread(maskpath, -1)[:,:,3], imsize)
    
    if cmd == "findvids" or cmd=="continue":
        vidfolder = "{}{}/videos/".format(datasets_path, dataset)
        kltfolder = "{}{}/klt/".format(datasets_path, dataset)
        mkdir(kltfolder)
        
        allvids = sorted(glob(vidfolder + "*.mkv"))
        
        if cmd == "continue":
            existing = sorted(glob(kltfolder + "*.pklz"))
            existing = [x.split('/')[-1].strip('.pklz') for x in existing]
            allvids = [x for x in allvids if not x.split('/')[-1].strip('.mkv') in existing]
            
        for vidpath in allvids:
            datpath = kltfolder + vidpath.split('/')[-1].replace('.mkv', '.pklz')
            outvidpath = datpath.replace('.pklz', '_klt.mp4')
            print_flush("{}   ->   {} & {}".format(vidpath, datpath, outvidpath))
            klt_save(vidpath, datpath, imsize, mask, outvidpath)
    else:
        raise(ValueError())
            
if __name__ == '__main__':
    main()
    
