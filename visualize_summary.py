""" A module for generating a summary video, which shows point tracks, 
    detections (in pixels and/or world coordinates), and tracks for
    parts of some randomly selected videos, to give an idea of how well things work.
    The different results are presented side-by-side in the video, and the
    results are included if available, otherwise not.
"""
    
import click
from glob import glob
from random import choice
from os.path import isfile

from folder import datasets_path, runs_path
from util import print_flush
from config import DatasetConfig

def get_klt_path(dataset_path, vid):
    return "{dsp}klt/{v}.pklz".format(dsp=dataset_path, v=vid)
    
def get_pixeldet_path(run_path, vid):
    return "{rp}csv/{v}.csv".format(rp=run_path, v=vid)
    
def get_worlddet_path(run_path, vid):
    return "{rp}detections_world/{v}_world.csv".format(rp=run_path, v=vid)

def get_worldtracks_path(run_path, vid):
    return "{rp}tracks_world/{v}_tracks.pklz".format(rp=run_path, v=vid)

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
       
    for vid in vids:
        f = get_klt_path(dataset_path, vid)
        if not isfile(f):
            include_klt = False
        
        f = get_pixeldet_path(run_path, vid)
        if not isfile(f):
            include_pixeldets = False
        
        f = get_worlddet_path(run_path, vid)
        if not isfile(f):
            include_worlddets = False

        f = get_worldtracks_path(run_path, vid)
        if not isfile(f):
            include_worldtracks = False
    
    print_flush("Point tracks: {}".format(include_klt))
    print_flush("Pixel coordinate detections: {}".format(include_pixeldets))
    print_flush("World coordinate detections: {}".format(include_worlddets))
    print_flush("World coordinate tracks: {}".format(include_worldtracks))
    
    # Decide where to start and stop in the videos
    dc = DatasetConfig(dataset)
    clip_length = clip_length*dc.get('video_fps') # convert from seconds to frames
    
    
    
if __name__ == '__main__':
    main()


    
    
