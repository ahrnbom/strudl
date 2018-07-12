""" A module for running a trained SSD on videos, saving the results as .csv files """

import os
from time import monotonic as time
from glob import glob
import click
import numpy as np
import imageio as io
import pandas as pd
import cv2
from math import floor, ceil
from keras.applications.imagenet_utils import preprocess_input

from video_imageio import get_model
from folder import mkdir, datasets_path, runs_path
from util import parse_resolution, print_flush
from apply_mask import Masker
from classnames import get_classnames
from ssd_utils import BBoxUtility

def rescale(df, index, factor):
    """ Rescales a data frame row, as integers. Used since detections are stored on scale 0-1 """
    s = df[index]
    s2 = [int(factor*x) for x in s]
    df[index] = s2      

def get_priors(model, input_shape):
    im_in = np.random.random((1,input_shape[1],input_shape[0],input_shape[2]))
    priors = model.predict(im_in,batch_size=1)[0, :, -8:]
    
    return priors

def generator(vid, input_shape, masker, batch_size):
    inputs = []
    old_inputs = []  

    for frame in vid:
        frame = masker.mask(frame)
        resized = cv2.resize(frame, (input_shape[0], input_shape[1]))
        inputs.append(resized)
        
        if len(inputs) == batch_size:
            inputs = np.array(inputs).astype(np.float64)
            inputs = preprocess_input(inputs)
            
            yield inputs
            
            old_inputs = inputs
            inputs = []       
    
    # Video finished, just keep repeating the same frame over and over
    # Keras expects infinite generators
    while True:
        yield old_inputs

def process_results(result, width, height, classnames, conf_thresh, step_size, frame_number):
    result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
    raw_detections = pd.DataFrame(np.vstack(result), columns=['class_index', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    
    rescale(raw_detections, 'xmin', width)
    rescale(raw_detections, 'xmax', width)
    rescale(raw_detections, 'ymin', height)
    rescale(raw_detections, 'ymax', height)
    rescale(raw_detections, 'class_index', 1)
           
    ci = raw_detections['class_index']
    cn = [classnames[int(x)-1] for x in ci]
    raw_detections['class_name'] = cn
    
    raw_detections['frame_number'] = (frame_number+1)
    
    return raw_detections[raw_detections.confidence>conf_thresh]

def generator_test(model, dataset, run, videopath, outname, classnames, input_shape, conf_thresh, priors, bbox_util, masker, batch_size, soft=False):
    with io.get_reader(videopath) as vid:
        gen = generator(vid, input_shape, masker, batch_size)
        
        width = input_shape[0]
        height = input_shape[1]
        
        all_detections = []
        step_size = int(ceil(len(vid)/batch_size))
        preds = model.predict_generator(gen, steps=step_size)
        
        all_detections = []        
        for i in range(len(vid)):
            pred = preds[i, :]
            pred = pred.reshape(1, pred.shape[0], pred.shape[1])
            results = bbox_util.detection_out(pred, soft=soft)

            detections = process_results(results, width, height, classnames, conf_thresh, step_size, i)
            all_detections.append(detections)
            
            if ((i+1)%500) == 0:
                print_flush(i+1)
                
        print_flush(i)
        dets = pd.concat(all_detections)
        dets.to_csv(outname) 


@click.command()
@click.option("--dataset", default="sweden2", help="Name of the dataset to use")
@click.option("--run", default="default", help="Run name of the SSD training run to use")
@click.option("--res", default="(640,480,3)", help="Image resolution that SSD was trained for, as a string on format '(width,height,channels)'")
@click.option("--conf", default=0.6, type=float, help="Confidence threshold of detections, as a float between 0 and 1")
@click.option("--bs", default=32, type=int, help="Batch size, the number of frames to feed into SSD in a batch, must fit on the GPU VRAM")
def detect(dataset, run, res, conf, bs):
    res = parse_resolution(res)

    vids = sorted(glob("{}{}/videos/*.mkv".format(datasets_path, dataset)))

    outfolder = "{}{}_{}/csv/".format(runs_path, dataset, run)
    mkdir(outfolder)

    classes = get_classnames(dataset)

    nvids = len(vids)

    input_shape = (res[0], res[1], 3)

    num_classes = len(classes)+1
    model = get_model(dataset, run, input_shape, num_classes)
    priors = get_priors(model, input_shape)
    bbox_util = BBoxUtility(num_classes, priors)
    masker = Masker(dataset)
    
    for i, vid in enumerate(vids):
        vname = vid.split('/')[-1]
        vsplit = vname.split('.')
        outname = outfolder + vsplit[0] + '.csv'
        
        if os.path.isfile(outname):
            print_flush("Skipping {}".format(outname))
            continue
        
        before = time()
  
        generator_test(model, dataset, run, vid, outname, classes, input_shape, conf, priors, bbox_util, masker, bs)
        
        done_percent = round(100*(i+1)/nvids)
        now = time()
        mins = floor((now-before)/60)
        secs = round(now-before-60*mins)
        print_flush("{}  {}% done, time: {} min {} seconds".format(vid, done_percent, mins, secs))
        
        
if __name__ == '__main__':
    detect()


