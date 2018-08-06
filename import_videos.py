""" Module for importing videos into a dataset project """

import click
import imageio as iio
from glob import glob
import cv2
import subprocess
from shutil import copy
import random
import string

from folder import mkdir, datasets_path
from util import parse_resolution, print_flush
from validation import validate_logfile
from timestamps import line_to_datetime

def fill(x, n):
    return str.zfill(str(x), n)

def generate_paths(time, target, logs_target, suffix):
    randstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    
    y = fill(time.year, 4)
    mo = fill(time.month, 2)
    d = fill(time.day, 2)
    h = fill(time.hour, 2)
    mi = fill(time.minute, 2)
    s = fill(time.second, 2)
    vidname = "{y}{mo}{d}_{h}{mi}{s}_{r}".format(y=y, mo=mo, d=d, h=h, mi=mi, s=s, r=randstring)
    
    vidpath = "{t}{vn}{sx}".format(t=target, vn=vidname, sx=suffix)
    logpath = "{lt}{vn}.log".format(lt=logs_target, vn=vidname)
    return vidpath, logpath
    
def read_log(logpath):
    with open(logpath, 'r') as f:
        lines = [x.rstrip() for x in f.readlines()]
    return lines

def recode_minutes_imageio(files, logs_basepath, minutes, width, height, fps, target, logs_target, suffix):
    """ Recodes videos such that each video is `minutes` many minutes long.
        Uses imageio to do this. Using handbrake would certainly be possible but a bit cumbersome to implement.
    """
    
    # Build a structure of the start times of each video, to sort them
    print_flush("Structuring...")
    vids = []
    for vid_path in files:
        video_name = '.'.join(vid_path.split('/')[-1].split('.')[0:-1])
        log_path = "{l}{vn}.log".format(l=logs_basepath, vn=video_name)
        
        with open(log_path, 'r') as f:
            first_line = f.readline().rstrip()
        
        first_time, frame_num = line_to_datetime(first_line)
        
        vids.append( (vid_path, log_path, first_time) )
    
    vids.sort(key = lambda x: x[2])
    
    # Go through the videos and build new videos, frame by frame
    can_make_more = True
    
    i_vid = 0
    i_frame = 0
    
    invid = iio.get_reader(vids[i_vid][0])
    inlog = read_log(vids[i_vid][1])
    
    rescale = True
    first_frame = invid.get_data(0)
    shape = first_frame.shape
    if (shape[0] == height) and (shape[1] == width):
        rescale = False
        print_flush("Does not resize")
    else:
        print_flush("Will resize to ({},{})".format(width, height))
                
    curr_time = vids[i_vid][2]
    
    while can_make_more:
        
        vidpath, logpath = generate_paths(curr_time, target, logs_target, suffix)
        print_flush("Making {}...".format(vidpath))
        
        outvid = iio.get_writer(vidpath, fps=fps)
        outlog = []
        
        first_time = curr_time
        
        while (curr_time - first_time).total_seconds()/60.0 < minutes:
            
            if i_frame >= len(inlog):
                # We need to jump to the next input video and log
                i_vid += 1
                i_frame = 0
                
                if i_vid >= len(vids):
                    can_make_more = False
                    break
                
                invid.close()
                invid = iio.get_reader(vids[i_vid][0])
                inlog = read_log(vids[i_vid][1])
            
            frame = invid.get_data(i_frame)
            line = inlog[i_frame]
            
            curr_time, _ = line_to_datetime(line)
            
            i_frame += 1
            
            if rescale:
                frame = cv2.resize(frame, (width, height))
            
            outvid.append_data(frame)
            outlog.append(line)
        
        # Close current output video/log
        outvid.close()
        
        with open(logpath, 'w') as f:
            for line in outlog:
                f.write("{}\n".format(line))

def encode_imageio(path, target_path, width, height, fps):
    rescale = True
    
    with iio.get_reader(path) as invid:
        with iio.get_writer(target_path, fps=fps) as outvid:
            for i,frame in enumerate(invid):
            
                # If resolution is the same, we should not rescale
                if i == 0:
                    shape = frame.shape
                    if (shape[0] == height) and (shape[1] == width):
                        rescale = False
                        print_flush("Does not resize")
            
                if rescale:
                    frame = cv2.resize(frame, (width, height))
                    
                outvid.append_data(frame)
                if (i+1)%500 == 0:
                    print_flush("  {}".format(i+1))
                       
def encode_handbrake(path, target_path, width, height, fps):
    cmd = ['HandBrakeCLI', '--width', str(width), 
           '--height', str(height), '--rate', str(fps),
           '-i', path, '-o', target_path]
        
    print_flush(' '.join(cmd))       
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    print_flush("  " + output)


@click.command()
@click.option("--query", help="A search query for finding video files, like '/media/someuser/somefolder/*.mkv'")
@click.option("--dataset", help="Name of the dataset to import the videos to")
@click.option("--resolution", help="New resolution. Use strings formatted like '(width,height,channels)'")
@click.option("--fps", type=int, help="Frames per second to encode to")
@click.option("--suffix", default='.mkv', help="Please use .mkv, as multiple script assume this")
@click.option("--method", default="imageio", help="Encoding/decoding library/method to use. Either 'imageio' or 'handbrake'. Handbrake mode assumes minutes to be 0.")
@click.option("--logs", help="Folder in which log files are stored")
@click.option("--minutes", default=0, help="If a positive integer, videos are recoded to videos of this many minutes in length. If 0, videos are kept to the previous length.")
def import_videos(query, dataset, resolution, fps, suffix, method, logs, minutes):   
    
    if method == "imageio":
        encode = encode_imageio
    elif method == "handbrake":
        encode = encode_handbrake
    else:
        raise(ValueError("Incorrect method {}".format(method)))
    
    resolution = parse_resolution(resolution)
    width, height = resolution[0:2]
    
    target = "{dp}{ds}/videos/".format(dp=datasets_path, ds=dataset)
    mkdir(target)
    
    logs_target = "{dp}{ds}/logs/".format(dp=datasets_path, ds=dataset)
    mkdir(logs_target)
    
    files = glob(query.strip("'"))
    files.sort()
    
    if minutes == 0:
        for path in files:
            video_name = '.'.join(path.split('/')[-1].split('.')[0:-1])
            
            target_path = "{t}{vn}{sx}".format(t=target, vn=video_name, sx=suffix)
            print_flush(target_path)
            
            encode(path, target_path, width, height, fps)
            
            src_log_path = "{l}{vn}.log".format(l=logs, vn=video_name)
            
            if validate_logfile(src_log_path):
                copy(src_log_path, logs_target)
                print_flush("Log file OK! {}".format(src_log_path))
            else:
                raise(ValueError("Incorrect log file {}".format(src_log_path)))
    else:
        assert(method == "imageio")
        recode_minutes_imageio(files, logs, minutes, width, height, fps, target, logs_target, suffix)
                    
    print_flush("Done!")       
    
if __name__ == '__main__':
    import_videos()


