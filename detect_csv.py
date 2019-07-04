""" A module for running a trained SSD on videos, saving the results as .csv files """

from time import monotonic as time
import click
import numpy as np
import imageio as io
import pandas as pd
import cv2
from math import floor, ceil
from keras.applications.imagenet_utils import preprocess_input
import subprocess
from subprocess import PIPE
import sys

from video_imageio import get_model
from folder import mkdir, datasets_path, runs_path
from util import parse_resolution, print_flush
from apply_mask import Masker
from classnames import get_classnames
from ssd_utils import BBoxUtility

python_path = sys.executable

def next_multiple(a, b):
    """ Finds a number that is equal to or larger than a, divisble by b """
    while not (a%b == 0):
        a += 1
    return a
    
def make_seqs(a, b):
    """ Makes mutliple sequences of length b, from 0 to a. The sequences are represented 
        as tuples with start and stop numbers. Note that the stop number of one sequence is
        the start number for the next one, meaning that the stop number is NOT included.
    """
    seqs = []
    x = 0
    while x < a:
        x2 = x + b
        if x2 > a:
            x2 = a
            
        seqs.append( (x, x2) )
        x = x2
    
    return seqs

def run_detector(dataset, run, videopath, outname, input_shape, conf_thresh, batch_size):
    
    
    with io.get_reader(videopath) as vid:        
        vlen = len(vid)
        
    vlen2 = next_multiple(vlen, batch_size)
    seq_len = next_multiple(1000, batch_size)
    
    # In the past, there was a memory leak that forced a division of the video into
    # shorted sequences. The memory leak was fixed, but this was kept because of
    # laziness.
    seqs = make_seqs(vlen2, seq_len)
    
    for i_seq,seq in enumerate(seqs):
        print_flush("From frame {} to {}...".format(seq[0], seq[1]))
        
        completed = subprocess.run([python_path, "detect_csv_sub.py", 
                         "--dataset={}".format(dataset),
                         "--run={}".format(run),
                         "--input_shape={}".format(input_shape),
                         "--seq_start={}".format(seq[0]),
                         "--seq_stop={}".format(seq[1]),
                         "--videopath={}".format(videopath),
                         "--conf_thresh={}".format(conf_thresh),
                         "--i_seq={}".format(i_seq),
                         "--outname={}".format(outname),
                         "--batch_size={}".format(batch_size)], stdout=PIPE, stderr=PIPE)
        if not (completed.returncode == 0):
            raise Exception("ERROR: Subprocess crashed. Return code: {}".format(completed.returncode))
        else:
            print_flush("Subprocess completed successfully")
        
        print_flush("Subprocess output:")    
        print_flush(completed.stdout.decode('UTF-8'))
        print_flush(completed.stderr.decode('UTF-8'))

@click.command()
@click.option("--dataset", default="sweden2", help="Name of the dataset to use")
@click.option("--run", default="default", help="Run name of the SSD training run to use")
@click.option("--res", default="(640,480,3)", help="Image resolution that SSD was trained for, as a string on format '(width,height,channels)'")
@click.option("--conf", default=0.6, type=float, help="Confidence threshold of detections, as a float between 0 and 1")
@click.option("--bs", default=32, type=int, help="Batch size, the number of frames to feed into SSD in a batch, must fit on the GPU VRAM")
@click.option("--clean", default=True, help="If True, all csv files are made from scratch. If False, any existing ones are kept")
def detect(dataset, run, res, conf, bs, clean):
    vids = list((datasets_path / dataset / "videos").glob('*.mkv'))
    vids.sort()

    outfolder = runs_path / "{}_{}".format(dataset,run) / "csv"
    mkdir(outfolder)

    nvids = len(vids)

    for i, vid in enumerate(vids):
        vname = vid.stem
        outname = outfolder / (vname+'.csv')
        
        if not clean:
            if outname.is_file():
                print_flush("Skipping {}".format(outname))
                continue
        
        before = time()
        
        print_flush(vname)
        run_detector(dataset, run, vid, outname, res, conf, bs)
        
        done_percent = round(100*(i+1)/nvids)
        now = time()
        mins = floor((now-before)/60)
        secs = round(now-before-60*mins)
        print_flush("{}  {}% done, time: {} min {} seconds".format(vid, done_percent, mins, secs))
    
    print_flush("Done!")
        
if __name__ == '__main__':
    detect()


