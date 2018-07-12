""" Module for importing videos into a dataset project """

import click
import imageio as io
from glob import glob
import cv2
import subprocess
from shutil import copy

from folder import mkdir, datasets_path
from util import parse_resolution, print_flush
from validation import validate_logfile

def encode_imageio(path, target_path, width, height, fps):
    rescale = True
    
    with io.get_reader(path) as invid:
        with io.get_writer(target_path, fps=fps) as outvid:
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
@click.option("--method", default="imageio", help="Encoding/decoding library/method to use. Either 'imageio' or 'handbrake'")
@click.option("--logs", help="Folder in which log files are stored")
def import_videos(query, dataset, resolution, fps, suffix, method, logs):   
    
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
                    
    print_flush("Done!")       
    
if __name__ == '__main__':
    import_videos()


