""" Module for preparing annotation of videos """

from random import random, shuffle
from glob import glob
import imageio as io
import numpy as np
from scipy.misc import imsave
import click

from folder import mkdir, datasets_path
from timestamps import Timestamps
from util import print_flush

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
    
def filtering(vidnames, nvids, ts):
    filtered = []
    for vid in vidnames:
        vidname = vid.split('/')[-1].strip('*.mkv')
        if random() < vidname_is_interesting(vidname, ts):
            filtered.append(vid)
    
    shuffle(filtered)
    return filtered[0:nvids]
    
def get_vidnames(dataset):
    search = "{}{}/videos/*.mkv".format(datasets_path, dataset)
    vidnames = glob(search)
    return vidnames
    
def gen_images(outbasepath, vidpath, n):
    folder = outbasepath + vidpath.split('/')[-1].strip('.mkv') + '/'
    mkdir(folder)
    
    with io.get_reader(vidpath) as vid:
        l = vid.get_length()
        fnums = np.linspace(0,l,n+2)
        fnums = [int(x) for x in fnums[1:n+1]]
        
        with open(folder + "frames.log", 'w') as f:
            f.write(vidpath.split('/')[-1] + "\n")
            for fn in fnums:
                f.write("{} ".format(fn))
        
        for i, fn in enumerate(fnums):
            frame = vid.get_data(fn)
            imsave(folder + "{}.jpg".format(i+1), frame)
            
def train_test_split(vidnames, train_amount):
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
def main(dataset, num_ims, ims_per_vid, train_amount):
    outbasepath = "{}{}/objects/".format(datasets_path, dataset)
    trainpath = outbasepath + "train/"
    testpath = outbasepath + "test/"
    
    ts = Timestamps(dataset)
    vidnames = filtering(get_vidnames(dataset), num_ims//ims_per_vid, ts)
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
    

    
