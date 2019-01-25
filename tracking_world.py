""" A module for performing tracking in world coordinates. This works better than
    tracking in pixel coordinates.
"""

import click
import pandas as pd
import numpy as np
from itertools import count
from math import sqrt
from munkres import Munkres, DISALLOWED, print_matrix, UnsolvableMatrix
from datetime import datetime
from random import choice
from copy import deepcopy
import json
from os.path import isfile

from world import Calibration
from util import parse_resolution, print_flush, pandas_loop, normalize, right_remove
from storage import load, save
from folder import datasets_path, runs_path, mkdir
from timestamps import Timestamps
from position import Check
from klt import Track

class WorldTrackingConfig(object):
    def __init__(self, conf_dict):
        self.conf_dict = conf_dict
        if not (self.validate()):
            raise(ValueError("Incorrect config dictionary"))
    
    def validate(self):
        conf_dict = self.conf_dict
    
        legal_keys = {'time_drop_thresh', 'time_region_check_thresh', 
                      'creation_too_close_thresh', 'is_too_close_thresh',
                      'incorrect_class_cost', 'cost_thresh', 'mask_margin',
                      'cost_dist_weight', 'cost_dir_weight'}
        legal_types = {int, float, dict}
        
        found_keys = set()
        
        for key in conf_dict:
            if not (key in legal_keys):
                return False
            
            val = conf_dict[key]
            if not (type(val) in legal_types):
                return False
            
            # Checking for legal subkeys here is a pain, so we skip doing that for now
            
            found_keys.add(key)
        
        if len(legal_keys) == len(found_keys): 
            return True
            
        return False
    
    def copy(self):
        return deepcopy(self)
    
    def __repr__(self):
        return "WorldTrackingConfig({})".format(self.conf_dict)
    
    def __str__(self):
        s = "World Tracking Config:\n"
        s += self.to_json()
        return s
    
    def to_json(self):
        return json.dumps(self.conf_dict, indent=2, sort_keys=True)
    
    def get_dict(self):
        # Want to "set" this? Just make a new config object
        return deepcopy(self.conf_dict)
        
    def keys(self):
        keys = []
    
        keys1 = list(self.conf_dict.keys())
        for key1 in keys1:
            if type(self.conf_dict[key1]) == dict:
                keys2 = list(self.conf_dict[key1].keys())
                for key2 in keys2:
                    keys.append( (key1, key2) )
            else:
                keys.append( (key1, None) )
        
        return keys
        
    def random_key(self):
        keys = self.keys()
        return choice(keys)
        
    def get(self, conf_key, class_name=None, class_name2=None):
        if conf_key in self.conf_dict:
            val = self.conf_dict[conf_key]
            
            if type(val) == dict:
                if class_name is None:
                    if 'default' in val:
                        return val['default']
                    else:
                        raise ValueError("Config key {} has no 'default' value, no class name provided".format(conf_key))
                else:
                    if not (class_name2 is None):
                        class_name += '_' + class_name2
                    
                    if class_name in val:
                        return val[class_name]
                    elif 'default' in val:
                        return val['default']
                    else:
                        raise ValueError("Config key {} has neither '{}' nor 'default'".format(conf_key, class_name))
            else:
                return val
                
        else:
            raise ValueError("No such world tracking config key: {}".format(conf_key))

""" How these world track configs work:
    It is a dict, with keys for the different parameters to configure. 
    If each key is mapped to a single value, that will always be used. 
    If multiple values are wanted, one for each class, then a dict should be provided, for example
        'some_setting': {'default': 5, 'bicycle': 10}
    The key 'default' will be used if the real class is not found.
    For some keys like interaction parameters, two classes are required. Then it should be like this:
        'some_setting': {'default': 5, 'car_bicycle': 10}
    Note that 'car_bicycle' is not the same as 'bicycle_car', so you may need to define both.
    
"""       
   
default_config = {
  "cost_dir_weight": 1.0147354311670724, # unitless
  "cost_dist_weight": 0.7620318412899034, # unitless
  "cost_thresh": {
    "bicycle": 20.304379446321803, # unitless? Compared with WorldTrack.cost output
    "default": 24.97096037057736 # unitless? Compared with WorldTrack.cost output
  },
  "creation_too_close_thresh": 2.712930167098239, # in meters
  "incorrect_class_cost": {
    "bicycle_person": 7.120520923545521, # unitless? Compared with WorldTrack.cost output
    "default": 5.485065801218582e+16, # unitless? Compared with WorldTrack.cost output
    "person_bicycle": 12.85008107267443 # unitless? Compared with WorldTrack.cost output
  },
  "is_too_close_thresh": {
    "bicycle_bicycle": 0.495536116573809, # in meters
    "default": 2.2825837807261196 # in meters
  },
  "mask_margin": 1.8100215477938848, # in pixels, how close to the borders of the interesting region a track can be
  "time_drop_thresh": 3.986878079720841, # in seconds
  "time_region_check_thresh": 0.3676807524471706 # in seconds
}

class WorldTrack(object):
    id_maker = count()
    
    def __init__(self, t, fn, x, y, dx, dy, cn, config):
        self.t = t   # time as datetime
        self.fn = fn # time as frame number
        self.t_since_update = t # this is not updated by auto_update so that track's drop time can be measured
        
        self.x = x # in m
        self.y = y # in m
        self.dx = dx # in m/s
        self.dy = dy # in m/s
        
        self.cn = cn # class name, as string
        
        # These are in m/s, computed from the track itself
        self.speed = 0
        self.smooth_speed = 0
        
        self.config = config
        
        # History contains tuples of the following format:
        # (frame number, datetime, world x, world y, world dx, world dy, speed (m/s), from_det)
        # where from_det is True if there is a detection at this position and False if it's just extrapolated
        self.history = []
        self.write_to_history()
        
        self.id = next(self.id_maker)
        
        self.drop_reason = 'not dropped yet'
        
    def update(self, t, fn, x, y, dx, dy):
        self.speed = sqrt( (x - self.x)**2 + (y - self.y)**2 ) / (t - self.t_since_update).total_seconds()
        self.smooth_speed = 0.9*self.smooth_speed + 0.1*self.speed
        
        self.t = t
        self.t_since_update = t
        self.fn = fn
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        
        self.write_to_history()
    
    def auto_update(self, now, frame_number):
        dt = (now - self.t).total_seconds()
    
        self.x += dt*self.dx
        self.y += dt*self.dy
        
        self.t = now
        self.fn = frame_number
        
        self.write_to_history(from_det=False)
        
    def write_to_history(self, from_det=True):
        self.history.append( (self.fn, self.t, self.x, self.y, self.dx, self.dy, self.smooth_speed, from_det) )
        
    def cost(self, t, x, y, dx, dy, cn):
        if cn == self.cn:
            class_cost = 0
        else:
            class_cost = self.config.get('incorrect_class_cost', self.cn, cn)

        dt = (t-self.t).total_seconds()
        
        x_cost = (self.x - x)**2
        y_cost = (self.y - y)**2
        
        dist_cost = (x_cost + y_cost)
        
        if self.smooth_speed > 0:
            dx_cost = abs(self.dx - dx)
            dy_cost = abs(self.dy - dy)
            
            dir_cost = (dx_cost + dy_cost)/self.smooth_speed # faster road users can't change their directions as much
        else:
            dir_cost = 0
            
        dist_weight = self.config.get('cost_dist_weight', self.cn)
        dir_weight = self.config.get('cost_dir_weight', self.cn)
        cost = class_cost + dist_weight*dist_cost + dir_weight*dir_cost
        return cost

def print_history(history):
    print_flush("frame    x      y   dx   dy    speed  fromdet")
    for h in history:
        line = ""
        for hh in h:
            if type(hh) == datetime:
                continue
            elif (type(hh) == float) or (type(hh) == np.float64):
                line += "  {:.2f}".format(hh)
            else:
                line += '  ' + str(hh)
            
        print_flush(line)

def lose_tracks(tracks, now, frame_number, mask_check, calib, config):
    indices_to_drop = set()
    for i, track in enumerate(tracks):    
    
        # Check if track is too old
        dt = (now - track.t_since_update).total_seconds()
        
        if dt > config.get('time_drop_thresh', track.cn):
            track.drop_reason = 'too old'
            indices_to_drop.add(i)
            continue
            
        if dt > config.get('time_region_check_thresh', track.cn):    
            # Check if track is too close to the 'restricted section'
            px, py = calib.to_pixels(track.x, track.y)
            if mask_check.test(px, py):
                track.drop_reason = 'too close to restricted section'
                indices_to_drop.add(i)
                continue
                
        for i_other, other_track in enumerate(tracks):
            if other_track == track:
                continue
                    
            if i_other in indices_to_drop:
                continue
            
            dx = track.x - other_track.x
            dy = track.y - other_track.y
            
            dist = sqrt(dx**2 + dy**2)
            
            if dist < config.get('is_too_close_thresh', track.cn, other_track.cn):
                # Which track to drop? Drop the younger one (a bit arbitrary)
                    
                if other_track.id > track.id:
                    other_track.drop_reason = 'too close to {}'.format(track.id)
                    indices_to_drop.add(i_other) 
                else:
                    track.drop_reason = 'too close to {}'.format(other_track.id)
                    indices_to_drop.add(i)
                    break
    
    just_lost = []
    indices_to_drop = list(indices_to_drop)
    indices_to_drop.sort()
    for i in reversed(indices_to_drop):
        just_lost.append(tracks.pop(i))
        
    return tracks, just_lost
    
def update_tracks(tracks, now, frame_number):
    for track in tracks:
        track.auto_update(now, frame_number)
        
    return tracks

def new_track(tracks, now, frame_number, det, config):
    """ Checks if it makes sense to make a new track, and if so, adds it to tracks """    

    ok = True
    for track in tracks:
        dist = sqrt((track.x-det['world_x'])**2 + (track.y-det['world_y'])**2)
        
        if dist < config.get('creation_too_close_thresh', det['class_name'], track.cn):
            ok = False
    
    if ok:
        track = WorldTrack(now, frame_number, det['world_x'], det['world_y'], det['world_dx'], det['world_dy'], det['class_name'], config)
        tracks.append(track)

    
def make_tracks(dataset, video_name, dets, klts, munkres, ts, calib, config, start_stop=None):
    """ Main function for making tracks in world coordinates.
    
        Arguments:
        dataset         -- name of dataset
        video_name      -- name of video (no folders or suffix)
        dets            -- world coordinate detections as made by detections_world.py
        klts            -- point tracks, as saved by detections_world.py (the 'per-detection point track format')
        munkres         -- a Munkres object (from the munkres module, not our code)
        ts              -- a Timestamps object (from the timestamps.py module)
        calib           -- a Calibration object (from the world.py module)
        config          -- a WorldTrackingConfig object (from this module)
        start_stop      -- either None of a tuple (start, stop) with integers of which frames to perform tracking on 
    """

    mask_check = Check(dataset, 'mask', margin=config.get('mask_margin'))

    tracks = []
    lost_tracks = []
    
    n_frames = max(dets['frame_number'])
    
    if start_stop is None:
        start_frame = 0
        stop_frame = n_frames
    else:
        start_frame, stop_frame = start_stop
    
    for frame_number in range(start_frame, stop_frame):  
        
        now = ts.get(video_name, frame_number)
        tracks, just_lost = lose_tracks(tracks, now, frame_number, mask_check, calib, config)
        lost_tracks.extend(just_lost)
        
        tracks = update_tracks(tracks, now, frame_number)
        
        dets_frame = dets[dets['frame_number'] == frame_number] # This is slow!
        
        if not tracks:
            # Let each detection be a track of its own
            for d in pandas_loop(dets_frame):
                track = new_track(tracks, now, frame_number, d, config)
                if not (track is None):
                    tracks.append(track)
                    
        else:
            # Hungarian algorithm to find associations           
            mat = []
            dets_list = [x for x in pandas_loop(dets_frame)]
            
            for i_track, track in enumerate(tracks):
                mat.append([])
                for i_det,det in enumerate(dets_list):       
                    cost = track.cost(now, det['world_x'], det['world_y'], 
                                      det['world_dx'], det['world_dy'], 
                                      det['class_name']) # this is slow!    
                    mat[i_track].append(cost)
              
            try:
                indices = munkres.compute(mat)
            except UnsolvableMatrix:
                # This means that tracks and detections were completely incompatible
                for d in pandas_loop(dets_frame):
                    new_track(tracks, now, frame_number, d, config)
                    
            else:
                for i_track, i_det in indices:
                    track = tracks[i_track]
                    if mat[i_track][i_det] <= config.get('cost_thresh', track.cn):
                        det = dets_list[i_det]
                        track.update(now, frame_number, 
                                     det['world_x'], det['world_y'], 
                                     det['world_dx'], det['world_dy'])
                                              
                        dets_list[i_det] = None # So that we can skip these when making new tracks
                
                for det in dets_list:
                    if det is None:
                        continue
                    
                    new_track(tracks, now, frame_number, det, config)
                    
    lost_tracks.extend(tracks)
    
    # Remove tracks that are too short to be considered reliable
    good_tracks = []
    for track in lost_tracks:
        from_det_count = 0
        for h in track.history:
            from_det = h[-1]
            if from_det:
                from_det_count += 1
        
        if from_det_count > 2:
            good_tracks.append(track)
    
    return good_tracks

@click.command()
@click.option("--cmd", default="findvids", help="Which command to run, 'findvids' to look for videos to track on, or else a name of a video (no folders or file suffix)")
@click.option("--dataset", default="sweden2", help="Which dataset to use")
@click.option("--run", default="default", help="Which training run to use")
@click.option("--conf", default=0.8, type=float, help="Confidence threshold")
@click.option("--make_videos", default=True, type=bool, help="If true, videos are generated. Can be slow")
def main(cmd, dataset, run, conf, make_videos):   
    if make_videos:
        from visualize_tracking import render_video
        from config import DatasetConfig
        from apply_mask import Masker
        
        mask = Masker(dataset)
        dc = DatasetConfig(dataset)
        
    config_path = "{rp}{ds}_{rn}/world_tracking_optimization.pklz".format(rp=runs_path, ds=dataset, rn=run)
    if isfile(config_path):
        config = load(config_path)
    else:
        #raise(ValueError("No world tracking optimized configuration exists at {}".format(config_path)))
        config = WorldTrackingConfig(default_config)
    
    calib = Calibration(dataset)    
    munkres = Munkres()
    ts = Timestamps(dataset)
    
    start_stop = None
    
    if cmd == "findvids":
        from glob import glob
        vidnames = glob('{dsp}{ds}/videos/*.mkv'.format(dsp=datasets_path, ds=dataset))
        vidnames = [right_remove(x.split('/')[-1], '.mkv') for x in vidnames]
        vidnames.sort()
        
        outfolder = '{}{}_{}/tracks_world/'.format(runs_path, dataset, run)
        mkdir(outfolder)
    else:
        vidnames = [cmd]
        outfolder = './'
        start_stop = (0,500)
            
    for v in vidnames:
        print_flush(v)    
        out_path = "{of}{v}_tracks.pklz".format(of=outfolder, v=v)
        
        print_flush("Loading data...")
        det_path = "{rp}{ds}_{rn}/detections_world/{v}_world.csv".format(rp=runs_path, ds=dataset, rn=run, v=v)
        detections3D = pd.read_csv(det_path)
        
        klt_path = det_path.replace('.csv', '_klt.pklz')
        klts = load(klt_path)
        
        print_flush("Tracking...")
        tracks = make_tracks(dataset, v, detections3D, klts, munkres, ts, calib, config, start_stop=start_stop)
        
        print_flush("Saving tracks...")
        save(tracks, out_path)
        
        if make_videos:

            vidpath = "{dsp}{ds}/videos/{v}.mkv".format(dsp=datasets_path, ds=dataset, v=v)
            print_flush("Rendering video...")
            render_video(tracks, vidpath, out_path.replace('.pklz','.mp4'), calib=calib, mask=mask, fps=dc.get('video_fps'))

    print_flush("Done!")
    
if __name__ == '__main__':
    main()
    


