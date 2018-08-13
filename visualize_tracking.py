import imageio as iio
import cv2
import numpy as np
from random import shuffle, choice
from bisect import bisect
import click
from os.path import isfile

from tracking import DetTrack
from tracking_world import WorldTrack, WorldTrackingConfig
from storage import load
from apply_mask import Masker
from world import Calibration
from folder import runs_path, datasets_path
from config import DatasetConfig
from util import print_flush

def get_colors(n=10):
    colors = []
    for i in range(0, n):
        # This can probably be written in a more elegant manner
        hue = 255*i/(n+1)
        col = np.zeros((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 180 # Saturation
        col[0][0][2] = 255 # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        colors.append(col)
        
    shuffle(colors) 
    return colors

"""
    Renders a video of a list of tracks.
     - tracks: List of DetTrack objects
     - vidpath: Path to a video file, to which tracks will be drawn on 
     - outvidname: Path to output video file
     - fps: Frames per second of output video
     - ncols: How many different colors should be used for different tracks
     - mask: Either None or a Masker object, which is applied to all frames before drawing (with alpha=0.5)
     - id_mode: "global" to show track IDs consistent with other videos from the same dataset, or "local" to make the first track in this video be ID 1
     - calib: If None, tracks are assumed to be in pixel coordinates. If a Calibration object (from the world.py module) then tracks are assumed to be in world coordinates and are projected back to pixel coordinates for this visualization
"""
def render_video(tracks, vidpath, outvidname, fps=10, ncols=50, mask=None, id_mode="global", calib=None):
    if id_mode == "global":
        pass # Keep track IDs as they are
    elif id_mode == "local":
        # Reset all track IDs, to be locally logical for this video
        i = 1
        # Sort by first appearance
        tracks = sorted(tracks, key= lambda x: x.history[0][0])
        for track in tracks:
            track.id = i
            i += 1

    colors = get_colors(ncols)

    with iio.get_reader(vidpath) as invid:
        with iio.get_writer(outvidname, fps=fps) as outvid:
            last_times = [x.history[-1][0] for x in tracks]
            n = max(last_times)    
            
            for i in range(1,n):
                frame = invid.get_data(i-1)
                if not (mask is None):
                    frame = mask.mask(frame, alpha=0.5)
                
                for track in tracks:
                    if calib is None:
                        draw(frame, track, i, colors[track.id%ncols])
                    else:
                        draw_world(frame, track, i, colors[track.id%ncols], calib)
                
                cv2.putText(frame, 'frame: {}'.format(i), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                outvid.append_data(frame)

def clamp(x, lower, upper):
    return sorted((lower, x, upper))[1]

def draw_world(to_draw, track, frame_number, color, calib):
    history = track.history
    fnums = [x[0] for x in history]
    if (frame_number >= fnums[0]) and (frame_number <= fnums[-1]):
        hist = history[bisect(fnums, frame_number)-1]
        x, y = hist[2:4]
        x, y = calib.to_pixels(x, y, 0)
        x, y = map(int, (x, y))
        text = "{} {}".format(track.cn, track.id)
        to_draw = _draw_world(to_draw, text, x, y, color)
        
def _draw_world(to_draw, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    text_top = (x, y-10)
    text_bot = (x + text_size[0][0]+10, y-5+text_size[0][1])
    text_pos = (x + 5, y-2)
    cv2.rectangle(to_draw, text_top, text_bot, color, -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)
    return to_draw
                
def draw(to_draw, track, frame_number, color):
    brightness = [1.8, 1.5, 1.2, 0.9, 0.7]
    
    hists = [hist for hist in track.history if hist[0] == frame_number]
    
    if not hists:
        return
    
    hist = None
    for loop_hist in hists:
        if loop_hist[5]:
            hist = loop_hist
    
    if hist is None:
        hist = choice(hists)

    bonustext = ""
    
    if frame_number in track.klt_checkpoints:
        klt_checkpoint = track.klt_checkpoints[frame_number]
        
        for i_klt, k in klt_checkpoint:
            kx = int(k[0])
            ky = int(k[1])
            bright = brightness[(i_klt%len(brightness))]
            klt_color = tuple([clamp(int(bright*c),0,255) for c in color])
            cv2.circle(to_draw, (kx, ky), 2, klt_color, -1)

    x = hist[1]
    y = hist[2]
    w = hist[3]
    h = hist[4]
    
    xmin = int(x - w/2)
    ymin = int(y - h/2)
    xmax = int(x + w/2)
    ymax = int(y + h/2)
    
    linewidth = 1
    
    if hist[5]:
        linewidth = 3
    
    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), color, linewidth)
       
    text = "{} {} {}".format(track.c, track.id, bonustext)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    text_size = cv2.getTextSize(text, font, font_scale, 1)
    text_top = (xmin, ymin-10)
    text_bot = (xmin + text_size[0][0]+10, ymin-5+text_size[0][1])
    text_pos = (xmin + 5, ymin-2)
    cv2.rectangle(to_draw, text_top, text_bot, color, -1)        
    cv2.putText(to_draw, text, text_pos, font, font_scale, (0,0,0), 1, cv2.LINE_AA)

@click.command()
@click.option("--dataset", type=str, help="Name of dataset")
@click.option("--run", type=str, help="Name of run")
@click.option("--videos", type=str, help="Either some videos separated by commas without spaces (or a single video), or, 'all' or 'random:X' where X is a number")
def main(dataset, run, videos):
    # Note: This main function only works for world coordinate tracks!
    
    calib = Calibration(dataset)
    dc = DatasetConfig(dataset)
    masker = Masker(dataset)
    
    if videos == 'all':
        from glob import glob
        files = glob('{rp}{ds}_{r}/tracks_world/*_tracks.pklz'.format(rp=runs_path, ds=dataset, r=run))
        video_names = [x.split('/')[-1].rstrip('_tracks.pklz') for x in files]
    elif videos.startswith('random:'):
        num = int(videos.lstrip('random:'))
        
        from glob import glob
        files = glob('{rp}{ds}_{r}/tracks_world/*_tracks.pklz'.format(rp=runs_path, ds=dataset, r=run))
        all_video_names = [x.split('/')[-1].rstrip('_tracks.pklz') for x in files]

        video_names = []        
        while len(video_names) < num:
            video_name = choice(all_video_names)
            if not video_name in video_names:
                video_names.append(video_name)
            
            # Just in case user wants more videos than there are
            if len(video_names) == len(all_video_names):
                break
        
    else:
        # Assumes the user types one or more videos, separated by commas with no spaces
        video_names = videos.split(',')
        
        # In case user includes endings
        video_names = [x.rstrip('.mkv') for x in video_names]
    
        # In case user includes spaces
        video_names = [x.strip(' ') for x in video_names]
        
    print_flush("Chosen videos: ")
    print_flush(str(video_names))
    for video_name in video_names:
        print_flush(video_name)
        print_flush("Loading...")
        tracks = load('{rp}{ds}_{r}/tracks_world/{v}_tracks.pklz'.format(rp=runs_path, ds=dataset, r=run, v=video_name))
        vidpath = "{dsp}{ds}/videos/{v}.mkv".format(dsp=datasets_path, ds=dataset, v=video_name)
        
        if not isfile(vidpath):
            raise(ValueError("Incorrect input {}".format(videos)))
        
        outvidpath = '{rp}{ds}_{r}/tracks_world/{v}_tracks.mp4'.format(rp=runs_path, ds=dataset, r=run, v=video_name)
        
        print_flush("Rendering...")
        render_video(tracks, vidpath, outvidpath, mask=masker, id_mode="global", calib=calib, fps=dc.get('video_fps'))
    
    print_flush("Done!")
        
if __name__ == '__main__':
    main()

