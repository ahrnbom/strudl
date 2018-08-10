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

from folder import datasets_path, runs_path
from util import print_flush
from config import DatasetConfig
from storage import load
from tracking_world import WorldTrack

def klt_frame(klt, frame):
    return None
    
def pixeldet_frame(det, frame):
    return None

def worlddet_frame(det, frame):
    return None
    
def worldtracks_frame(tracks, frame):
    return None

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
       
    for vid in vids:
        f = get_klt_path(dataset_path, vid)
        if not isfile(f):
            include_klt = False
        else:
            klts.append(load(f))
        
        f = get_pixeldet_path(run_path, vid)
        if not isfile(f):
            include_pixeldets = False
        else:
            pixeldets.append(pd.read_csv(f))
        
        f = get_worlddet_path(run_path, vid)
        if not isfile(f):
            include_worlddets = False
        else:
            worlddets.append(pd.read_csv(f))

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
    dc = DatasetConfig(dataset)
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

    for i_vid, vid in enumerate(vids):
        with iio.get_reader("{dsp}videos/{v}.mkv".format(dsp=dataset_path, v=vid)) as invid:
            for i_frame, frame in enumerate(invid):
                
                pieces = []
                
                for inc, fun, dat in zip(incs, funs, dats):
                    if inc:
                        pieces.append(fun(dat[i_frame], frame))
                
                
                return                
                
                
                
    
if __name__ == '__main__':
    main()


    
    
