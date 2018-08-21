""" A module for optimizing tracking in world coordinates. Note that this is 
    often not necessary as the default parameters tend to work fine. 
"""

import pandas as pd
import numpy as np
from datetime import datetime
from munkres import Munkres
from math import sqrt
from bisect import bisect
from random import uniform 
import click
import json

from world import Calibration
from util import pandas_loop, print_flush, split_lambda
from timestamps import Timestamps
from folder import datasets_path, runs_path
from position import Check
from tracking_world import make_tracks, WorldTrackingConfig
from storage import load, save
from plot import simple_plot
from config import DatasetConfig

def visualize_tracks(outvidpath, dataset, gts, tracks=None, stack_axis='v'):
    import imageio as iio
    from visualize_tracking import _draw_world, draw_world
    from visualize import class_colors
    from apply_mask import Masker
    from config import DatasetConfig
    
    if not (tracks is None):
        calib = Calibration(dataset)
        
        # Reset IDs
        tracks = sorted(tracks, key= lambda x: x.history[0][0])
        for track in tracks:
            track.id = i
            i += 1
    
    dc = DatasetConfig(dataset)
    
    gts_by_vid = split_lambda(gts, lambda x: x[0])
    assert(len(gts_by_vid) == 1)
    vid = list(gts_by_vid.keys())[0]
    
    n_colors = 50
    colors = class_colors(n_colors)
    
    mask = Masker(dataset)
    
    with iio.get_writer(outvidpath, fps=dc.get('video_fps')) as outvid:
        with iio.get_reader("{dsp}{ds}/videos/{v}.mkv".format(dsp=datasets_path, ds=dataset, v=vid)) as invid:
            
            gt_by_frame = split_lambda(gts, lambda x: x[1])
            fns = list(gt_by_frame.keys())
            fns.sort()
            
            for fn in fns:
                gts_frame = gt_by_frame[fn]
                
                frame = invid.get_data(fn)
                frame = mask.mask(frame, alpha=0.5)
                
                if not (tracks is None):
                    tracks_frame = frame.copy()
                
                for gt in gts_frame:
                    vid, fn, t, x, y, i, c, px, py = gt

                    text = "{} {}".format(c,i)
                    col = colors[i%n_colors]
                    
                    frame = _draw_world(frame, text, px, py, col)
                
                if not (tracks is None):
                    for track in tracks:
                        draw_world(tracks_frame, track, fn, colors[track.id%n_colors], calib)
                    
                    if stack_axis == 'h':
                        frame = np.hstack( (frame, tracks_frame) )
                    elif stack_axis == 'v':
                        frame = np.vstack( (frame, tracks_frame) )
                    else:
                        raise(ValueError("Incorrect stack axis {}, try 'h' or 'v'".format(stack_axis)))
                              
                outvid.append_data(frame)

def interpret_tracks_gt(dataset, date, det_id, traj_csv_path):
    """ Interprets tracking ground truth csv files exported by T-Analyst.
    
        Parameters:
        dataset         -- name of dataset
        date            -- date when the video was filmed, as a string on format 'YYYY-MM-DD'
        det_id          -- the ID number of the T-Analyst 'detection' of interest. Set to None to include everthing in the .csv file
        traj_csv_path   -- path to .csv file exported by T-Analyst
    """
    
    traj = pd.read_csv(traj_csv_path, sep=';',decimal=',')

    calib = Calibration(dataset)
    ts = Timestamps(dataset)
    mask = Check(dataset, 'mask')

    gts = []

    for traj_row in pandas_loop(traj):
        row_det_id = traj_row['Detection ID']
        if row_det_id == det_id:
            c = traj_row['Type of road user']
            i = traj_row['Road user ID']
            x = traj_row['X (m)']
            y = traj_row['Y (m)']
            t = traj_row['Time Stamp']
            
            #t = date + ' ' + t
            #t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            # strptime is both slow and has issues with the way milliseconds are written by T-Analyst
            year, month, day = map(int, date.split('-'))
            hour,minute,second,millisecond = map(int, t.replace('.',':').split(':'))
            t = datetime(year, month, day, hour, minute, second, millisecond*1000)
            
            vid, fn = ts.get_frame_number(t)

            px, py = calib.to_pixels(x,y)
            px, py = map(int, (px,py))
            
            if not mask.test(px,py):    
                gt = (vid, fn, t, x, y, i, c, px, py)
                gts.append(gt)
    
    return gts

def compute_cost(gt_track, track, gt_class_name_conversion, ts, vid):
    """ This should be based on these factors:
         - The time overlap. Low cost if they start and end at around the same frame numbers.
         - The positional distance at each frame in the ground truth track. Low cost if they are close. A standard cost of say 64 m distance should be applied when track does not exist at this time.
         - Classes. This should be 0 if they are the same class, and a large number if they are not the same class
    """
    
    gt_class = gt_track[0][6]
    t_class = track.cn
    
    if not (gt_class_name_conversion is None):
        if gt_class in gt_class_name_conversion:
            gt_class = gt_class_name_conversion[gt_class]
    
    class_cost = 0
    if not (gt_class == t_class):
        class_cost = 9001
    
    t_start = ts.get(vid, track.history[0][0])
    t_end = ts.get(vid, track.history[-1][0])
    
    gt_framenums = [x[1] for x in gt_track]
    gt_start = ts.get(vid, min(gt_framenums))
    gt_end = ts.get(vid, max(gt_framenums))
    
    dt_start = abs( (gt_start - t_start).total_seconds() )
    dt_end = abs( (gt_end - t_end).total_seconds() )
    
    time_cost = (dt_start**2 + dt_end**2)
    
    dist_cost = 0
    for gt in gt_track:
        fn = gt[1]
        gt_x, gt_y = gt[3:5]
        
        track_fnums = [x[0] for x in track.history]
        if (fn >= track_fnums[0]) and (fn <= track_fnums[-1]):
            hist = track.history[bisect(track_fnums, fn)-1]
            track_x, track_y = hist[2:4]
            dist = sqrt( (gt_x - track_x)**2 + (gt_y - track_y)**2 )
        else:
            dist = 64
        
        dist_cost += dist
            
    dist_cost /= len(gt_framenums)
    
    return class_cost + time_cost + dist_cost
    
def score_tracking(dataset, run, gt, tracking_config, gt_class_name_conversion):
    munkres = Munkres()
    ts = Timestamps(dataset)
    calib = Calibration(dataset)
    
    # Each video separately
    by_vid = split_lambda(gt, lambda x: x[0])
    all_costs = {}
    all_tracks = {}
    
    for vid in by_vid:
        gt_list = by_vid[vid]
        
        fn = [x[1] for x in gt_list]
        start_stop = (min(fn), max(fn))
        
        gt_tracks = split_lambda(gt_list, lambda x: x[5], as_list=True)
        
        print_flush("  Loading data...")
        det_path = "{rp}{ds}_{rn}/detections_world/{v}_world.csv".format(rp=runs_path, ds=dataset, rn=run, v=vid)
        detections3D = pd.read_csv(det_path)
        
        klt_path = det_path.replace('.csv', '_klt.pklz')
        klts = load(klt_path)
        
        print_flush("  Tracking...")
        tracks = make_tracks(dataset, vid, detections3D, klts, munkres, ts, calib, tracking_config)
        all_tracks[vid] = tracks
        
        # Associate each track with a ground truth one, based on cost.
        # Then compute total cost and use as score measure
        
        print_flush("  Associating tracks with ground truth...")
        mat = []
        for igt, gt_track in enumerate(gt_tracks):
            mat.append([])
            for it, track in enumerate(tracks):
                cost = compute_cost(gt_track, track, gt_class_name_conversion, ts, vid)
                mat[igt].append(cost)
        
        try:
            indices = munkres.compute(mat)
        except UnsolvableMatrix:
            cost_sum = float("inf")
        else:
            print_flush("  Computing cost...")
            cost_sum = 0
            for igt, it in indices:
                cost_sum += mat[igt][it]
        
        all_costs[vid] = cost_sum
    
    if len(by_vid) == 1:    
        vid = list(by_vid.keys())[0]
        return all_costs[vid], all_tracks[vid]
    else:
        raise(ValueError("Computing scores for multiple videos, while supported in this function, is not generally supported in this module. Remove this exception from the code if you want to add multi-video scoring support. But then you need to take care of the output as dictionaries"))
        
        return all_costs, all_tracks

def random_config(config, config_min, config_max, key1, key2=None):
    keys = [(key1, key2)]
    if key1 == 'all':
        keys = config.keys()
    
    val = None
    for key in keys:
        key1, key2 = key
        
        if key2 is None:
            val_min = config_min.conf_dict[key1]
            val_max = config_max.conf_dict[key1]
        else:
            val_min = config_min.conf_dict[key1][key2]
            val_max = config_max.conf_dict[key1][key2]
        
        val = uniform(val_min, val_max)
        
        if key2 is None:
            config.conf_dict[key1] = val
        else:
            config.conf_dict[key1][key2] = val
        
    return config, val
    

def optimize_tracking(config_min, config_max, dataset, run, gt, gt_class_name_conversion, n=4, patience=20, plot_path=None):
    """ Black box optimization strategy: 
        1. Start by randomly choosing parameter values within their intervals.
        2. Compute current score, and save these parameters as current best
        3. Choose a random parameter, and give it 'n' random values within the interval.
           Compute the score for each. Take the best of these n+1 options (including the 'current best')
        4. Repeat step 3 until score hasn't improved for 'patience' number of runs
    """
    
    best_config, _ = random_config(config_min.copy(), config_min, config_max, 'all')
    best_cost, best_tracks = score_tracking(dataset, run, gt, best_config, gt_class_name_conversion)
    print_flush("Initial cost: {}".format(best_cost))
    print_flush(best_config)
    print_flush('')
    
    counter = 0
    t = 0
    ts = [t]
    cs = [best_cost]
    
    while counter < patience:
        counter += 1
        t += 1
        
        key1, key2 = config_min.random_key()
        key_str = key1
        if not (key2 is None):
            key_str += ':' + key2
        print_flush("Optimizing over {}...".format(key_str))
        
        for i in range(n):
            config, val = random_config(best_config, config_min, config_max, key1, key2)
            print_flush("Trying {} = {}".format(key_str, val))
            cost, tracks = score_tracking(dataset, run, gt, config, gt_class_name_conversion)
            
            if cost < best_cost:
                counter = 0
                best_cost = cost
                best_config = config
                best_tracks = tracks
                
                print_flush("New best cost {}".format(cost))
                print_flush(best_config)
                print_flush('') 
        
        ts.append(t)
        cs.append(best_cost)
        
        if not (plot_path is None):
            print_flush("Plotting...")
            simple_plot(ts, cs, filepath = plot_path, xlabel="Time (iterations)", ylabel="Cost")
        
        print_flush("Finished optimizing over {}".format(key_str))
        print_flush("Iterations without improvement: {} / {}".format(counter, patience))
        print_flush('')
    
    print_flush("Optimal config found")
    print_flush(best_config)
    print_flush("")
    print_flush("Done!")
    
    return best_config, best_tracks

@click.command()
@click.option("--dataset", help="Name of dataset")
@click.option("--run", help="Name of training run")
@click.option("--date", help="Date when the ground truth trajectories exist, e.g. '2017-05-16' as a string") 
@click.option("--gt_csv", help="Full path to a ground truth trajectory .csv file exported by T-Analyst")
@click.option("--det_id", type=int, help="The T-Analyst 'detection' ID number for which ground truth trajectories exist")
@click.option("--gt_class_name_conversion", default=None, help="A dictionary table in JSON for converting class names between trajectory csv and class names in STRUDL, e.g. '{\"pedestrian\": \"person\"}'. Can also be None to just keep all classes")
@click.option("--visualize", type=bool, help="If true, a visualization video is made of the best tracking result")
@click.option("--patience", default=20, help="How many iterations can go without lowering the cost, before stopping")
@click.option("--per_iteration", default=4, help="How many random values are tested in each iteration")
def main(dataset, run, date, gt_csv, det_id, gt_class_name_conversion, visualize, patience, per_iteration):
    
    dc = DatasetConfig(dataset)
    vidres = dc.get('video_resolution')
    width, height, _ = vidres
    if width > height:
        stack_axis = 'v'
    else:
        stack_axis = 'h'
        
    if gt_class_name_conversion is None:
        print_flush("Not converting class names")    
    else:
        print_flush("Using class conversion:")
        print_flush(gt_class_name_conversion)
        gt_class_name_conversion = json.loads(gt_class_name_conversion)
        assert(type(gt_class_name_conversion) == dict)
    
    print_flush("Interpreting ground truth...")
    gt = interpret_tracks_gt(dataset, date, det_id, gt_csv)

    print_flush("Optimizing...")
    
    config_min = {
    'time_drop_thresh': 0.1, # in seconds
    'time_region_check_thresh': 0.1, # in seconds
    'creation_too_close_thresh': 1, # in meters
    'is_too_close_thresh': {'default': 0.2, 'bicycle_bicycle': 0.1}, # in metres
    'incorrect_class_cost': {'default': 100, 'bicycle_person': 3,
                             'person_bicycle': 3}, # unitless? Compared with WorldTrack.cost output
    'cost_thresh': {'default':5, 'bicycle':5}, # unitless? Compared with WorldTrack.cost output
    'mask_margin': 0, # in pixels, how close to the borders of the interesting region a track can be
    'cost_dist_weight': 0.5,
    'cost_dir_weight': 0.5,
                 }
    
    config_max = {
    'time_drop_thresh': 7.0, # in seconds
    'time_region_check_thresh': 2.0, # in seconds
    'creation_too_close_thresh': 10, # in meters
    'is_too_close_thresh': {'default': 3.0, 'bicycle_bicycle': 2.0}, # in metres
    'incorrect_class_cost': {'default': 123456789123456789, 'bicycle_person': 30,
                             'person_bicycle': 30}, # unitless? Compared with WorldTrack.cost output
    'cost_thresh': {'default':25, 'bicycle':35}, # unitless? Compared with WorldTrack.cost output
    'mask_margin': 15, # in pixels, how close to the borders of the interesting region a track can be
    'cost_dist_weight': 2.0,
    'cost_dir_weight': 2.0,
                 }
                     
    config_min, config_max = map(WorldTrackingConfig, (config_min, config_max))
    
    base_path = "{rp}{ds}_{r}/world_tracking_optimization".format(rp=runs_path, ds=dataset, r=run)
    plot_path = base_path + '.png'
    
    config, tracks = optimize_tracking(config_min, config_max, dataset, run, gt, 
                                       gt_class_name_conversion, plot_path=plot_path, 
                                       patience=patience, n=per_iteration)
                                       
    save(config, base_path + '.pklz')
    
    if visualize:
        print_flush("Visualizing...")
        visualize_tracks(base_path + '.mp4', dataset, gt, tracks, stack_axis=stack_axis)
        
    print_flush("Done!")

if __name__ == '__main__':
    main()


