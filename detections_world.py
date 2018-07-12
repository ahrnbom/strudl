""" Module for converting detections to world coordinates, including KLT-based speed measurement """

import click
import pandas as pd
import numpy as np
from itertools import count

from world import Calibration
from util import parse_resolution, print_flush, pandas_loop, normalize
from storage import load, save
from folder import datasets_path, runs_path, mkdir
from timestamps import Timestamps
from tracking import convert_klt, find_klt_for_frame
from classnames import get_classnames
from config import DatasetConfig

class PointTrackStructure(object):
    def __init__(self, klts, klt_frames, width, height, nx=16, ny=16):
        self.klt_frames = {}
        for index in klt_frames:
            self.klt_frames[index] = set(klt_frames[index])
            
        self.klts = klts
        
        self.nx = nx
        self.ny = ny
        
        self.w2 = int(width/(nx-1))
        self.h2 = int(height/(ny-1))
        
        self.data = []
        for iy in range(self.ny):
            self.data.append([])
            for ix in range(self.nx):
                self.data[iy].append(set())
        
        for i, klt in enumerate(self.klts):
            if 'taken' in klt:
                klt.pop('taken')
                
            for key in klt:
                if type(key) == int:
                    x, y = klt[key]
                    x, y = map(int, (x, y))
                    klt[key] = (x, y)
                    
                    ix = int(x)//self.w2
                    iy = int(y)//self.h2

                    if not i in self.data[iy][ix]:
                        self.data[iy][ix].add(i)
    
    def get_klts(self, frame_number, det):
        frame_indices = self.klt_frames[frame_number]
    
        xstart = int(det['xmin']//self.w2)
        xstop = int(det['xmax']//self.w2) + 1 # because range doesn't go all the way up
        ystart = int(det['ymin']//self.h2)
        ystop = int(det['ymax']//self.h2) + 1
        
        found = set()
        
        for iy in range(ystart, ystop):
            for ix in range(xstart, xstop):
                pos_indices = self.data[iy][ix]
                indices = frame_indices.intersection(pos_indices)
                
                found.update(indices)
        
        return [self.klts[x] for x in found]

def detections_to_3D(dets, pts, calib, ts, v, klt_save_path=None):
    """ Treat each detection like a point with a direction """
    
    cx = (dets['xmin'] + dets['xmax'])//2
    cy = (dets['ymin'] + dets['ymax'])//2
    
    dets['cx'] = cx
    dets['cy'] = cy
    
    world_x = []
    world_y = []
    
    for px, py in zip(cx, cy):
        x, y, z = calib.to_world(px, py)
        world_x.append(x)
        world_y.append(y)
    
    dets['world_x'] = world_x
    dets['world_y'] = world_y
    
    # Compute approximate motion direction for each detection, using KLT tracks and transforming the direction to world coordinates
    wdxs = []
    wdys = []
    
    id_maker = count()
    ids = []
    
    all_matching_klts = {}
    
    for det in pandas_loop(dets):
        det_id = next(id_maker)
        ids.append(det_id)
        
        fn = det['frame_number']
        klts_frame = pts.get_klts(fn, det)
        
        dx = 0
        dy = 0
        n = 0
        
        klt_matches = []
        
        for k in klts_frame:
            x, y = k[fn]
            
            # Compute average speed in m/s
            if (x > det['xmin']) and (x < det['xmax']) and (y > det['ymin']) and (y < det['ymax']): 
                previous = (x,y)
                previous_fn = fn
                if (fn-1) in k:
                    previous_fn = fn-1
                    previous = k[previous_fn]
                    
                
                following = (x,y)
                following_fn = fn
                if (fn+1) in k:
                    following_fn = fn+1
                    following = k[following_fn]
                
                dt = (ts.get(v, following_fn) - ts.get(v, previous_fn)).total_seconds()
                if dt > 0:
                    # dx and dy are here in pixels/second
                    dx += (following[0]-previous[0])/dt
                    dy += (following[1]-previous[1])/dt
                    n += 1
                    
                    klt_matches.append(k)
                
        if ((abs(dx) > 0) or (abs(dy) > 0)) and (n > 0):
            # Average speed in pixels/second
            dx /= n
            dy /= n
            
            wx2, wy2, _ = calib.to_world(det['cx'] + dx, det['cy'] + dy)
            wdx = wx2 - det['world_x']
            wdy = wy2 - det['world_y']
            
            # These should now be in m/s
            wdxs.append(wdx)
            wdys.append(wdy)
            
        else: 
            wdxs.append(0)
            wdys.append(0)
        
        all_matching_klts[det_id] = klt_matches
    
    dets['world_dx'] = wdxs
    dets['world_dy'] = wdys
    dets['id'] = ids
    
    if not (klt_save_path is None):
        save(all_matching_klts, klt_save_path)
           
    return dets

@click.command()
@click.option("--cmd", default="findvids", help="Which command to run, 'findvids' to look for videos to track on, or else a name of a video (no folders or file suffix)")
@click.option("--dataset", default="sweden2", help="Which dataset to use")
@click.option("--run", default="default", help="Which training run to use")
@click.option("--vidres", default="(640,480,3)", help="Resolution of the videos, like '(640,480,3)' with width, height and then number of channels")
@click.option("--ssdres", default="(640,480,3)", help="Resolution images fed into object detector, like '(640,480,3)' with width, height and then number of channels")
@click.option("--kltres", default="(320,240)", help="Resolution of images used for point tracking, like '(320, 240)' with width and then height")
@click.option("--make_videos", default=True, type=bool, help="If true, videos are generated. Can be slow")    
def main(cmd, dataset, run, vidres, ssdres, kltres, make_videos):
    vidres = parse_resolution(vidres)
    ssdres = parse_resolution(ssdres)
    kltres = parse_resolution(kltres)
    
    x_factor = float(vidres[0])/ssdres[0]
    y_factor = float(vidres[1])/ssdres[1]
    det_dims = ('xmin', 'xmax', 'ymin', 'ymax')
    det_factors = (x_factor, x_factor, y_factor, y_factor)

    calib = Calibration(dataset)
    ts = Timestamps(dataset)
    
    class KLTConfig(object):
        klt_x_factor = 0
        klt_y_factor = 0
        
    klt_config = KLTConfig()
    klt_config.klt_x_factor = vidres[0]/kltres[0]
    klt_config.klt_y_factor = vidres[1]/kltres[1]
    
    if cmd == "findvids":
        from glob import glob
        vidnames = glob('{dsp}{ds}/videos/*.mkv'.format(dsp=datasets_path, ds=dataset))
        vidnames = [x.split('/')[-1].strip('.mkv') for x in vidnames]
        vidnames.sort()
        
        outfolder = '{}{}_{}/detections_world/'.format(runs_path, dataset, run)
        mkdir(outfolder)
    else:
        vidnames = [cmd]
        outfolder = './'
        
    mkdir(outfolder)

    if make_videos:
            classnames = get_classnames(dataset)
            dc = DatasetConfig(dataset)
            fps = dc.get('video_fps')
    
    for v in vidnames:
        print_flush(v) 
        detections = pd.read_csv('{}{}_{}/csv/{}.csv'.format(runs_path, dataset, run, v))
            
        # Convert pixel coordinate positions from SSD resolution to video resolution
        # because Calibration assumes video resolution coordinates
        for dim, factor in zip(det_dims, det_factors):
            detections[dim] = round(detections[dim]*factor).astype(int)
        
        print_flush("Converting point tracks...")    
        klt = load('{}{}/klt/{}.pklz'.format(datasets_path, dataset, v))
        klt, klt_frames = convert_klt(klt, klt_config)
        pts = PointTrackStructure(klt, klt_frames, vidres[0], vidres[1])
        
        outpath = '{of}{v}_world.csv'.format(of=outfolder, v=v)
        
        print_flush("Converting to world coordinates...")
        detections3D = detections_to_3D(detections, pts, calib, ts, v, klt_save_path=outpath.replace('.csv', '_klt.pklz'))
        
        detections3D.to_csv(outpath, float_format='%.4f')
        
        if make_videos:
            from visualize_detections import detections_video
            vidpath = "{dsp}{ds}/videos/{v}.mkv".format(dsp=datasets_path, ds=dataset, v=v)
            
            print_flush("Rendering video...")
            detections_video(detections3D, vidpath, outpath.replace('.csv', '.mp4'), classnames, dataset, vidres, fps=fps, conf_thresh=0.0, coords='world')
    
    print_flush("Done!")
            
if __name__ == '__main__':
    main()


