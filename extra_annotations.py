""" A module for adding additional annotations based on specific times """

import click
from random import random, shuffle
import imageio as iio
import numpy as np
from scipy.misc import imsave
from datetime import datetime

from folder import mkdir, datasets_path
from timestamps import Timestamps
from util import print_flush, right_remove
from config import DatasetConfig

@click.command()
@click.option("--dataset", type=str, help="Name of dataset")
@click.option("--times", type=str, help="Comma-separated list of timestamps. Images will be taken from two-second intervals around these timestamps. On format '2017-05-16 00:49:04.954000'")
@click.option("--images_per_time", type=int, default=10, help="Number of images to gather for each timestamp")
@click.option("--interval", type=float, default=2.0, help="How long time in seconds around each timestamp to collect images from")
def main(dataset, times, images_per_time, interval):
    times = times.strip("'") # These are added around it by the quote function in server.py, to make sure it is a single argument instead of being split by the spaces
    
    ts = Timestamps(dataset)
    
    dc = DatasetConfig(dataset)
    fps = dc.get('video_fps')
    half_interval = int((fps*interval)/2) # in frames
    
    timestrings = times.split(',')
    for timestring in timestrings:
        print_flush(timestring)
        
        # Intepret the requested times, can look like '2017-05-16 00:49:04.954000'
        splot = timestring.split(' ')
        date = splot[0].split('-')
        time = splot[1].replace('.',':').split(':')
        
        year,month,day = map(int, date)
        hour,minute,second,microsecond = map(int, time)
        
        timestamp = datetime(year,month,day,hour,minute,second,microsecond)
        vid_name, frame_num = ts.get_frame_number(timestamp)
        
        print_flush("Time found to be {}, frame {}".format(vid_name, frame_num))
        
        if vid_name is None:
            raise(ValueError("This timestamp was incorrect: {} Could it be before the first video?".format(timestring)))
        
        video_path = datasets_path / dataset / "videos" / (vid_name+'.mkv')
        
        annot_folder = datasets_path / dataset / "objects" / "train" / vid_name
        log_path = annot_folder / 'frames.log'
        if not log_path.is_file():
            with log_path.open('w') as f:
                f.write("{}.mkv\n".format(vid_name))
        
        # See which frames were already annotated, to start at the right index
        already_ims = list(annot_folder.glob('*.jpg'))
        if already_ims:
            already_nums = [int(x.stem) for x in already_ims]
            i = max(already_nums) + 1
        else:
            i = 1
        
        with iio.get_reader(video_path) as vid:
            # Find start and end time, in frames
            start = frame_num - half_interval
            if start < 0:
                start = 0
            
            stop = frame_num + half_interval                   
            if stop >= len(vid):
                stop = len(vid)-1
            
            with open(log_path, 'a') as log:
                
                # Choose frames to extract 
                frame_nums = np.linspace(start, stop, images_per_time).astype(int).tolist()
                frame_nums = sorted(list(set(frame_nums))) # Remove duplicates
                
                for frame_num in frame_nums:
                    frame = vid.get_data(frame_num)
                    
                    log.write("{} ".format(frame_num))
                    
                    impath = annot_folder / "{}.jpg".format(i)
                    imsave(impath, frame)
                    
                    i += 1
                    
                    print_flush("> Written {}".format(impath))
                    
    print_flush("Done!")
    
if __name__ == '__main__':
    main()


                
    
    
