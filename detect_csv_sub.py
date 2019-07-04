""" Because detect_csv.py used to leak memory, part of it can run as a subprocess 
    by running this module
"""

import click
import numpy as np
import imageio as io
import pandas as pd
import cv2
from math import floor, ceil
from keras.applications.imagenet_utils import preprocess_input

from video_imageio import get_model
from util import parse_resolution, print_flush
from apply_mask import Masker
from classnames import get_classnames
from ssd_utils import BBoxUtility

def get_priors(model, input_shape):
    im_in = np.random.random((1,input_shape[1],input_shape[0],input_shape[2]))
    priors = model.predict(im_in,batch_size=1)[0, :, -8:]
    
    return priors

def rescale(df, index, factor):
    """ Rescales a data frame row, as integers. Used since detections are stored on scale 0-1 """
    s = df[index]
    s2 = [int(factor*x) for x in s]
    df[index] = s2      

def process_results(result, width, height, classnames, conf_thresh, frame_number):
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
    
    raw_detections['frame_number'] = frame_number
    
    dets = raw_detections[raw_detections.confidence>conf_thresh]
    del raw_detections
    return dets

@click.command()
@click.option("--dataset", type=str, help="Name of dataset")
@click.option("--run", type=str, help="Name of run")
@click.option("--input_shape", default="(640,480,3)", help="CNN input resolution")
@click.option("--seq_start", type=int, help="Frame to start with (included)")
@click.option("--seq_stop", type=int, help="Frame to stop at (not included)")
@click.option("--videopath", type=str, help="Path to input video")
@click.option("--conf_thresh", type=float, help="Confidence threshold")
@click.option("--i_seq", type=int, help="0-based number of when in sequence this is")
@click.option("--outname", type=str, help="Path to output csv file")
@click.option("--batch_size", type=int, help="Batch size for executing neural network")
def main(dataset, run, input_shape, seq_start, seq_stop, videopath, conf_thresh, i_seq, outname, batch_size):
    
    print_flush("> Predicting...")
    classes = get_classnames(dataset)
    masker = Masker(dataset)
    
    input_shape = parse_resolution(input_shape)
    
    num_classes = len(classes)+1
    model = get_model(dataset, run, input_shape, num_classes, verbose=False)
    priors = get_priors(model, input_shape)
    bbox_util = BBoxUtility(num_classes, priors)
    
    
    width = input_shape[0]
    height = input_shape[1]
    
    inputs = []
    outputs = []
    old_frame = None
    
    with io.get_reader(videopath) as vid: 
        vlen = len(vid)
        for i_in_seq in range(seq_start, seq_stop):
            if i_in_seq < vlen:
                frame = vid.get_data(i_in_seq)
                frame = masker.mask(frame)
                old_frame = frame
            else:
                frame = old_frame
                
            resized = cv2.resize(frame, (width, height))
            inputs.append(resized)
            
            if len(inputs) == batch_size:
                inputs2 = np.array(inputs)
                inputs2 = inputs2.astype(np.float32)
                inputs2 = preprocess_input(inputs2)
                
                y = model.predict_on_batch(inputs2)
                outputs.append(y)
                
                inputs = []     
        
    preds = np.vstack(outputs)
    
    print_flush("> Processing...")
    all_detections = []   
    seq_len = seq_stop - seq_start
         
    for i in range(seq_len):
        frame_num = i + seq_start
        
        if frame_num < vlen:           
            pred = preds[i, :]
            pred = pred.reshape(1, pred.shape[0], pred.shape[1])
            results = bbox_util.detection_out(pred, soft=False)

            detections = process_results(results, width, height, classes, conf_thresh, frame_num)
            all_detections.append(detections)
    
    dets = pd.concat(all_detections)
    
    # For the first line, we should open in write mode, and then in append mode
    # This way, we still overwrite the files if this script is run multiple times
    open_mode = 'a'
    include_header = False
    if i_seq == 0:
        open_mode = 'w'
        include_header = True

    print_flush("> Writing to {} ...".format(outname))    
    with open(outname, open_mode) as f:
        dets.to_csv(f, header=include_header) 


if __name__ == "__main__":
    main()
