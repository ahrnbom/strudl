""" Module for preparing annotation of videos, extracting images to annotate, chosen
    in a reasonably intelligent manner.
"""

from random import random, shuffle
from pathlib import Path
import imageio as io
import numpy as np
from scipy.misc import imsave
import click

from folder import mkdir, datasets_path
from timestamps import Timestamps
from util import print_flush, right_remove

def vidname_is_interesting(vidname, ts):
    # Determines if a videoname sounds interesting or not. Used to make sure we don't 
    # annotate too much at night, when barely anything interesting happens
    # Outputs the probability that the video will pass through the intial filtering
    
    t = ts.get(vidname)
    hour = t.hour
    
    # night is not very interesting
    if hour >= 22 or hour <= 4:
        return 0.2
    
    return 1.0
    
def filtering(vidnames, nvids, ts, night):
    """ Gets nvids many random videos, possibly with lower probability of night """
    filtered = []
    for vid in vidnames:
        vidname = vid.stem
        if night:
            if random() < vidname_is_interesting(vidname, ts):
                filtered.append(vid)
        else:
            filtered.append(vid)
    
    shuffle(filtered)
    return filtered[0:nvids]
    
def get_vidnames(dataset):
    return list((datasets_path / dataset / "videos").glob('*.mkv'))
    
def gen_images(outbasepath, vidpath, n):
    """ Pick n images evenly spread out over the video """
    
    folder = outbasepath / vidpath.stem
    mkdir(folder)
    
    with io.get_reader(vidpath) as vid:
        l = vid.get_length()
        
        # Avoid the edges of the video
        fnums = np.linspace(0,l,n+2) 
        fnums = [int(x) for x in fnums[1:n+1]]
        
        # Log files allow these to be recreated, if necessary.
        # These logs are used when mining rare classes, to avoid annotating to close to existing annotations
        with (folder / "frames.log").open('w') as f:
            f.write(vidpath.stem + "\n")
            for fn in fnums:
                f.write("{} ".format(fn))
        
        for i, fn in enumerate(fnums):
            frame = vid.get_data(fn)
            imsave(folder / "{}.jpg".format(i+1), frame)
            
def train_test_split(vidnames, train_amount):
    """ Splits the dataset into train and test set.
        At the time of writing this, test set is not used for anything.
    """
    n = len(vidnames)
    n_train = int(n*train_amount)
    n_test = n-n_train
    
    shuffle(vidnames)
    train = vidnames[0:n_train]
    test = vidnames[n_train:n]
    return train, test

@click.command()
@click.option("--dataset", default="sweden2", help="Name of the dataset to annotate")
@click.option("--num_ims", default=500, help="Number of images to annotate, in total")
@click.option("--ims_per_vid", default=20, help="How many images per video to annotate")
@click.option("--train_amount", default=1.0, help="How many of the images that should be part of the training and validation sets, as a float between 0 and 1. The rest will be in a test set")
@click.option("--night", default=True, type=bool, help="If True, fewer night videos will be included. If False, all videos are treated equally")
def main(dataset, num_ims, ims_per_vid, train_amount, night):
    outbasepath = datasets_path / dataset / "objects"
    trainpath = outbasepath / "train"
    testpath = outbasepath / "test"
    
    ts = Timestamps(dataset)
    vidnames = filtering(get_vidnames(dataset), num_ims//ims_per_vid, ts, night)
    train, test = train_test_split(vidnames, train_amount)
    
    print_flush("Train:")
    for v in train:
        print_flush(v)
        gen_images(trainpath, v, ims_per_vid)
    
    print_flush("Test:")
    for v in test:
        print_flush(v)
        gen_images(testpath, v, ims_per_vid)
        
    print_flush("Done!")
    
if __name__ == "__main__":
    main()
    

    
