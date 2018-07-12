""" A module for taking object detections as CSV files, and visualizing them on 
    top of videos
"""

import cv2
import imageio as io
import numpy as np
from glob import glob
import pandas as pd
import click
from math import ceil

from visualize import draw, class_colors
from apply_mask import Masker
from classnames import get_classnames
from util import parse_resolution, print_flush
from folder import mkdir, datasets_path, runs_path
from world import Calibration

def make_divisible(x, y):
    return int(y*ceil(float(x)/y))

def detections_video(detections, videopath, outvideopath, classnames, dataset, res, fps=15, conf_thresh=0.75, show_frame_number=True, coords='pixels'):
    """ Renders a video with the detections drawn on top
    
    Arguments:
    detections        -- the detections as a pandas table
    videopath         -- path to input video
    outvideopath      -- path to output video showing the detections
    classnames        -- list of all the classes
    dataset           -- name of the dataset
    res               -- resolution of output video and coordinates in csv file (assumed to be the same). Probably SSD resolution if performed on direct csv files, and probably the video resolution if performed on csv files with world coordinates
    fps               -- frames-per-second of output video
    conf_thresh       -- Detections with confidences below this are not shown in output video. Set to negative to not visualize confidences, or set to 0.0 to show all of them.   
    show_frame_number -- writes the frame number in the top left corner of the video
    coords            -- coordinate system of detections
    """
    
    masker = Masker(dataset)
    
    calib = None
    if coords == 'world':
        calib = Calibration(dataset)

    num_classes = len(classnames)+1
    colors = class_colors(num_classes)

    outwidth = make_divisible(res[0], 16)
    outheight = make_divisible(res[1], 16)
    pad_vid = True
    if (outwidth == res[0]) and (outheight == res[1]):
        pad_vid = False
    
    with io.get_reader(videopath) as vid:
        with io.get_writer(outvideopath, fps=fps) as outvid:
            for i,frame in enumerate(vid):
                frame = masker.mask(frame, alpha=0.5)
                frame = cv2.resize(frame, (res[0], res[1]))
                
                dets = detections[detections['frame_number']==i]
                if len(dets) > 0:
                    frame = draw(frame, dets, colors, conf_thresh=conf_thresh, coords=coords, calib=calib)
                
                if pad_vid:
                    padded = 255*np.ones((outheight, outwidth, 3), dtype=np.uint8)
                    padded[0:res[1], 0:res[0], :] = frame
                    frame = padded    
                
                if show_frame_number:
                    cv2.putText(frame, 'Frame {}'.format(i), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                outvid.append_data(frame)
                
                
@click.command()
@click.option("--cmd", default="findvids", help="Either 'findvids' to search for videos, or a path to a specific video's csv file containing detections")
@click.option("--res", default="(640,480,3)", help="Resolution that the detections are in, on the format '(width,height,channels)'. If working with pixel coordinates, then this should be the detector's resolution. If world coordinates, it should be video resolution")
@click.option("--dataset", default="sweden2", help="Name of the dataset")
@click.option("--run", default="default", help="Name of training run")
@click.option("--conf", default=0.0, type=float, help="Confidence threshold")
@click.option("--fps", default=15, type=int, help="Frames-per-second of output video")
@click.option("--coords", default="pixels", type=click.Choice(['pixels', 'world']), help="Coordinate system of data in csv files ('pixels' or 'world')")
def main(cmd, res, dataset, run, conf, fps, coords):
    res = parse_resolution(res)
    classnames = get_classnames(dataset)
    
    local_output = False
    csvs = []
    if cmd == "findvids":
        if coords == "pixels":
            query = "{rp}{ds}_{r}/csv/*.csv".format(rp=runs_path, ds=dataset, r=run)
        elif coords == "world":
            query = "{rp}{ds}_{r}/detections_world/*.csv".format(rp=runs_path, ds=dataset, r=run)
            
        found = glob(query)
        found.sort()
        csvs.extend(found)
    else:
        csvs.append(cmd)
        local_output = True
    
    if coords == "pixels":
        out_folder = '{rp}{ds}_{r}/detections/'.format(rp=runs_path, ds=dataset, r=run)
    elif coords == "world":
        out_folder = '{rp}{ds}_{r}/detections_world/'.format(rp=runs_path, ds=dataset, r=run)
        
    mkdir(out_folder)
    
    for csv_path in csvs:
        vidname = csv_path.split('/')[-1].strip('.csv')
        if coords == "world":
            vidname = vidname.strip('_world')
            
        vid_path = "{dsp}{ds}/videos/{v}.mkv".format(dsp=datasets_path, ds=dataset, v=vidname)

        if local_output:
            outvid_path = '{}.mp4'.format(vidname)
        else:
            outvid_path = '{}{}.mp4'.format(out_folder, vidname)        
        
        detections = pd.read_csv(csv_path)
        detections_video(detections, vid_path, outvid_path, classnames, dataset, res, fps=fps, conf_thresh=conf, coords=coords)
        print_flush(outvid_path)

if __name__ == '__main__':
    main()


