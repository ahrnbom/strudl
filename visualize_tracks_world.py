""" A module for visualizing tracks in both pixel and world coordinates in a video """

import imageio as iio
import cv2
import numpy as np
import click
from bisect import bisect
from os.path import isfile

from tracking import DetTrack
from storage import load
from apply_mask import Masker
from world import Calibration
from folder import runs_path, datasets_path
from config import DatasetConfig
from visualize_tracking import get_colors
from util import print_flush

def draw_world(to_draw, track, hist, frame_number, color, calib, scale):
    im_h, im_w, im_c = to_draw.shape
    
    x, y, w, h = hist[1:5]
    
    x, y, z = calib.to_world(x, y)
    
    #TODO
    X0 = -51.6198188452061
    Y0 = -11.6329304603976
    Dx = 0.927839172679427
    Dy = -0.372980521799136
    Scale = 7.63256930074245
    
    x = int(Scale*(Dx*x - X0)) #int(im_w*(x-scale[0])/(scale[1]-scale[0]))
    y = int(Scale*(Dy*y - Y0)) #int(im_h*(y-scale[2])/(scale[3]-scale[2]))
    
    text = "{} {}".format(track.c, track.id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    text_top = (x, y-10)
    text_bot = (x + text_size[0][0]+10, y-5+text_size[0][1])
    text_pos = (x + 5, y-2)
    cv2.rectangle(to_draw, text_top, text_bot, color, -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)
    
def draw(to_draw, track, hist, frame_number, color):    
    x, y, w, h = hist[1:5]
    
    xmin = int(x - w/2)
    ymin = int(y - h/2)
    xmax = int(x + w/2)
    ymax = int(y + h/2)
    
    linewidth = 1
    
    if hist[5]:
        linewidth = 3
    
    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), color, linewidth)
       
    text = "{} {}".format(track.c, track.id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    text_top = (xmin, ymin-10)
    text_bot = (xmin + text_size[0][0]+10, ymin-5+text_size[0][1])
    text_pos = (xmin + 5, ymin-2)
    cv2.rectangle(to_draw, text_top, text_bot, color, -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)
            
            
@click.command()
@click.option("--dataset", help="Name of dataset")
@click.option("--run", help="Name of training run")
@click.option("--video_name", help="Name of video, without file suffix or folder names")
@click.option("--out_path", default="tracking_on_map.mp4", help="Path to output video file, should be in .mp4 format")
def render(dataset, run, video_name, out_path):
    ncols = 50
        
    dc = DatasetConfig(dataset)
    mask = Masker(dataset)
    colors = get_colors(ncols)
    calib = Calibration(dataset)
    
    tracks = load("{rp}{ds}_{r}/tracks/{vn}_tracks.pklz".format(rp=runs_path, ds=dataset, r=run, vn=video_name))
    vid_path = "{dsp}{ds}/videos/{vn}.mkv".format(dsp=datasets_path, ds=dataset, vn=video_name)
    
    map_path = "{dsp}{ds}/map.png".format(dsp=datasets_path, ds=dataset)
    if isfile(map_path):
        map_im = cv2.imread(map_path)
    else:
        map_im = None
    
    # Reset all track IDs, to be locally logical for this video
    i = 1
    # Sort by first appearance
    tracks = sorted(tracks, key= lambda x: x.history[0][0])
    for track in tracks:
        track.id = i
        i += 1
    
    # Find the "edges" of the world in world coordinates
    wminx = float("inf")
    wmaxx = -float("inf")
    wminy = float("inf")
    wmaxy = -float("inf")
    
    for track in tracks:
        for h in track.history:
            x, y = h[1:3]
            wx, wy, wz = calib.to_world(x, y)
            if wx < wminx:
                wminx = wx
            if wx > wmaxx:
                wmaxx = wx
            if wy < wminy:
                wminy = wy
            if wy > wmaxy:
                wmaxy = wy
    
    scale = (wminx, wmaxx, wminy, wmaxy)
    
    with iio.get_writer(out_path, fps=dc.get('video_fps')) as outvid:
        with iio.get_reader(vid_path) as invid:
            for i, frame in enumerate(invid):
                if not (mask is None):
                    frame = mask.mask(frame, alpha=0.5)
                
                if map_im is None:
                    world_frame = 100*np.ones_like(frame)
                else:
                    world_frame = map_im.copy()
                
                for track in tracks:
                    col = colors[track.id%ncols]
                    history = track.history
                    fnums = [x[0] for x in history]
                    if (i >= fnums[0]) and (i <= fnums[-1]):
                        hist = history[bisect(fnums, i)-1]
                    
                        draw(frame, track, hist, i, col)
                        draw_world(world_frame, track, hist, i, col, calib, scale)
                
                new_frame = np.concatenate((frame, world_frame), axis=0)
                
                outvid.append_data(new_frame)
                
                if i%500 == 0:
                    print_flush(i)
    
if __name__ == '__main__':
    render()


