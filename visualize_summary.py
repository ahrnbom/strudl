""" A module for generating a summary video, which shows point tracks, 
    detections (in pixels and/or world coordinates), and tracks for
    parts of some randomly selected videos, to give an idea of how well things work.
    The different results are presented side-by-side in the video, and the
    results are included if available, otherwise not.
"""
    
import click
from glob import glob
from random import choice, randint
from os.path import isfile
import imageio as iio
import pandas as pd
import numpy as np
import cv2

from folder import datasets_path, runs_path
from util import print_flush, pandas_loop
from config import DatasetConfig, RunConfig
from storage import load
from tracking_world import WorldTrack
from tracking import convert_klt, find_klt_for_frame
from visualize import class_colors, draw
from apply_mask import Masker
from classnames import get_classnames

def join(pieces):
    """ Joins some images into a single one, stracking the best it can.
        Works with up to four images.
    """
    
    if len(pieces) == 1:
        return pieces[0]
    elif len(pieces) == 2:
        return np.vstack(pieces)
    else:
        shape = pieces[0].shape
        new_shape = (shape[0]*2, shape[1]*2, shape[2])
        
        new_im = np.zeros(new_shape, dtype=np.uint8)
        xs = [0, shape[1], 0,        shape[1]]
        ys = [0, 0,        shape[0], shape[0]]
        
        for piece, x, y in zip(pieces, xs, ys):
            new_im[y:y+shape[0], x:x+shape[1], :] = piece
        
        return new_im

def klt_frame(data, frame, i_frame):
    klts, klt_frames, colors = data

    klts = find_klt_for_frame(klts, klt_frames, i_frame)

    for klt in klts:
        klt_id = klt['id']
        x, y = map(int, klt[i_frame])
        
        color = colors[klt_id % 50]
        
        cv2.circle(frame, (x, y), 2, color, -1)
    
    return frame
    
def pixeldet_frame(data, frame, i_frame):
    dets, colors, x_scale, y_scale = data
    det = dets[dets['frame_number'] == i_frame]
    
    frame = draw(frame, det, colors, x_scale=x_scale, y_scale=y_scale, coords='pixels')

    return frame

def worlddet_frame(det, frame, i_frame):
    return frame
    
def worldtracks_frame(tracks, frame, i_frame):
    return frame

def get_klt_path(dataset_path, vid):
    return "{dsp}klt/{v}.pklz".format(dsp=dataset_path, v=vid)
    
def get_pixeldet_path(run_path, vid):
    return "{rp}csv/{v}.csv".format(rp=run_path, v=vid)
    
def get_worlddet_path(run_path, vid):
    return "{rp}detections_world/{v}_world.csv".format(rp=run_path, v=vid)

def get_worldtracks_path(run_path, vid):
    return "{rp}tracks_world/{v}_tracks.pklz".format(rp=run_path, v=vid)

def make_clip(vid, clip_length, dataset_path):
    log = "{dsp}logs/{v}.log".format(dsp=dataset_path, v=vid)
    
    with open(log, 'r') as f:
        lines = f.readlines()
    
    l = len(lines)
    del lines
    
    start = randint(1, l-clip_length-1)
    stop = start + clip_length
    
    return start, stop
    
    

@click.command()
@click.option("--dataset", type=str, help="Name of dataset")
@click.option("--run", type=str, help="Name of run")
@click.option("--n_clips", default=4, help="Maximum number of videos to take clips from")
@click.option("--clip_length", default=60, help="Length of clips, in seconds")
def main(dataset, run, n_clips, clip_length):
    dc = DatasetConfig(dataset)
    rc = RunConfig(dataset, run)
    mask = Masker(dataset)
    classes = get_classnames(dataset)
    num_classes = len(classes)+1
    
    dataset_path = "{dsp}{ds}/".format(dsp=datasets_path, ds=dataset)
    run_path = "{rp}{ds}_{r}/".format(rp=runs_path, ds=dataset, r=run)
    
    # Grab a bunch of videos
    vids_query = "{dsp}videos/*.mkv".format(dsp=dataset_path)
    all_vids = glob(vids_query)
    all_vids = [x.split('/')[-1].rstrip('.mkv') for x in all_vids]
    
    all_vids.sort()
    
    vids = []
    
    if n_clips > len(all_vids):
        n_clips = len(all_vids)
        
    if n_clips == len(all_vids):
        vids = all_vids
    else:
        while len(vids) < n_clips:
            vid = choice(all_vids)
            if not vid in vids:
                vids.append(vid)
    
    print_flush(vids)
           
    # Find out what has been run on all of these videos, what to include
    include_klt = True
    include_pixeldets = True
    include_worlddets = True
    include_worldtracks = True
    
    klts = []
    pixeldets = []
    worlddets = []
    worldtracks = []
       
    # Point tracks need to be converted for faster access
    vidres = dc.get('video_resolution')
    kltres = dc.get('point_track_resolution')
       
    class KLTConfig(object):
        klt_x_factor = 0
        klt_y_factor = 0
        
    klt_config = KLTConfig()
    klt_config.klt_x_factor = vidres[0]/kltres[0]
    klt_config.klt_y_factor = vidres[1]/kltres[1]
    
    ssdres = rc.get('detector_resolution')
    x_scale = vidres[0]/ssdres[0]
    y_scale = vidres[1]/ssdres[1]
       
    for vid in vids:
        f = get_klt_path(dataset_path, vid)
        if not isfile(f):
            include_klt = False
        else:
            klt = load(f)
            klt, klt_frames = convert_klt(klt, klt_config)
            pts = (klt, klt_frames, class_colors(50))
            klts.append(pts)
        
        f = get_pixeldet_path(run_path, vid)
        if not isfile(f):
            include_pixeldets = False
        else:
            dets = pd.read_csv(f)
                        
            pixeldets.append( (dets, class_colors(num_classes), x_scale, y_scale) )
        
        f = get_worlddet_path(run_path, vid)
        if not isfile(f):
            include_worlddets = False
        else:
            dets = pd.read_csv(f)
                
            worlddets.append(dets)

        f = get_worldtracks_path(run_path, vid)
        if not isfile(f):
            include_worldtracks = False
        else:
            worldtracks.append(load(f))
    
    print_flush("Point tracks: {}".format(include_klt))
    print_flush("Pixel coordinate detections: {}".format(include_pixeldets))
    print_flush("World coordinate detections: {}".format(include_worlddets))
    print_flush("World coordinate tracks: {}".format(include_worldtracks))
    
    # Decide where to start and stop in the videos
    clip_length = clip_length*dc.get('video_fps') # convert from seconds to frames
    
    print_flush("Clip length in frames: {}".format(clip_length))
    
    clips = []
    for vid in vids:
        start, stop = make_clip(vid, clip_length, dataset_path)
        clips.append( (start, stop) )
    
    incs = [include_klt, include_pixeldets, include_worlddets, include_worldtracks]
    funs = [klt_frame, pixeldet_frame, worlddet_frame, worldtracks_frame]
    dats = [klts, pixeldets, worlddets, worldtracks]
    
    print_flush(clips)
        
    with iio.get_writer("{dsp}summary.mp4".format(dsp=dataset_path), fps=dc.get('video_fps')) as outvid:
        for i_vid, vid in enumerate(vids):
            old_prog = 0
            
            with iio.get_reader("{dsp}videos/{v}.mkv".format(dsp=dataset_path, v=vid)) as invid:
                start, stop = clips[i_vid]
                for i_frame in range(start, stop):
                    frame = invid.get_data(i_frame)
                    
                    pieces = []
                    
                    for inc, fun, dat in zip(incs, funs, dats):
                        if inc:
                            pieces.append(fun(dat[i_vid], mask.mask(frame.copy(), alpha=0.5), i_frame))
                    
                    outvid.append_data(join(pieces))
                    
                    prog = float(i_frame-start)/(stop-start)
                    if prog-old_prog > 0.1:
                        print_flush("{}%".format(round(prog*100)))
                        old_prog = prog    
                
                
                
    
if __name__ == '__main__':
    main()


    
    