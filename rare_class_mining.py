"""
    A script for finding additional images to annotate that probably contains 
    rarely occuring object classes.
"""

import click
import datetime
from datetime import timedelta
import imageio as iio
import cv2
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

from folder import datasets_path
from timestamps import Timestamps
from util import parse_resolution, print_flush, pandas_loop, rep_last
from apply_mask import Masker
from classnames import get_classnames
from train import train

def process(inputs, frame_nums, im_origs, vids, confidence, class_name, soft, batch_size2, model, bbox_util, classes):
    found_data = []
    
    inputs = np.array(inputs).astype(np.float64)
    inputs = preprocess_input(inputs)
    
    preds = model.predict(inputs, batch_size=batch_size2, verbose=0)
    results = bbox_util.detection_out(preds, soft=soft)
    
    for result, frame_num, im_res, v in zip(results, frame_nums, im_origs, vids):
        result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
        for r in result:
            if r[1] > confidence:
                this_class_name = classes[int(r[0])-1]
                if this_class_name == class_name:
                    found_data.append( (v, frame_num, im_res) )
                    print_flush("Found an object of class {} in frame {} in video {}".format(class_name,frame_num,v.stem))

                    # Once we've found an object of the right class, we don't care about this image any more
                    break
                        
    return found_data

@click.command()
@click.option("--dataset", default="default", help="Name of dataset")
@click.option("--class_name", default="bus", help="Name of object class to look for")
@click.option("--time_dist", default=10.0, help="How many seconds away from other annotations and each other the selected frames may be.")
@click.option("--sampling_rate", default=5.0, help="How often all videos should be sampled to get images to be searched, in seconds.")
@click.option("--import_datasets", default="", type=str, help="Additional datasets to include during training, separated by commas")
@click.option('--input_shape',default='(300,300,3)',help='The size into which the images are rescaled before going into SSD')
@click.option('--image_shape', default='(300,300,3)',help='The size of the original training images')
@click.option('--batch_size', default=4, help='The batch size of the training.')
@click.option('--batch_size2', default=32, help='The batch size of the testing.')
@click.option('--epochs', default=75, help='The number of epochs used to run the training.')
@click.option('--frozen_layers', default=3, help='The number of frozen layers, max 5.')
@click.option('--confidence', default=0.3, help='The minimum confidence of a detected object of the given class (can be 0)')
def rare_class_mining(dataset, class_name, time_dist, sampling_rate, import_datasets, input_shape, image_shape, batch_size, batch_size2, epochs, frozen_layers, confidence):
    soft = False
    classes = get_classnames(dataset)
    
    ts = Timestamps(dataset)

    # Find all videos in dataset
    vidnames = list((datasets_path / dataset / "videos").glob('*.mkv'))
    
    all_found = []
    
    for v in vidnames:
        
        # Find video length from log file (computing this from the video file is too slow)
        log_file = (datasets_path / dataset / "logs" / v.with_suffix('.log').name).read_text().split('\n')
        last = -1
        while not log_file[last]:
            last -= 1
        last_line = log_file[last]
        v_len = int(last_line.split(' ')[0])
        
        print_flush("{} of length {}".format(v, v_len))
        
        # Find existing annotations
        frames_log = (datasets_path / dataset / "objects" / "train" / v.stem / "frames.log").read_text().split()
        frames_log = [x for x in frames_log[1:] if x] # Remove first line, which is video name, and any empty lines
        annotated = [int(x) for x in frames_log]
        print_flush("Avoiding the following existing frames: ")
        print_flush(str(annotated))
        
        curr_time = ts.get(v.stem, 0)

        annotated_times = [ts.get(v.stem, x) for x in annotated]
        
        found = []
        found_times = []
        done = False
                
        while not done:
            # Sample in time
            curr_time += timedelta(seconds=sampling_rate)            
            curr_frame = ts.get_frame_number_given_vidname(curr_time, v.stem)
            
            if curr_frame >= v_len:
                # We have reached the end of the video
                done = True
                continue
            
            if curr_frame in annotated:
                continue
                
            # Check if we are too close to existing annotations
            dists = [abs((curr_time-x).total_seconds()) for x in annotated_times]
            if any([(x <= time_dist) for x in dists]):
                continue
                        
            # Check if we are too close to any previously chosen interesting frames
            dists = [abs((curr_time-x).total_seconds()) for x in found_times]
            if any([(x <= time_dist) for x in dists]):
                continue
                                
            # This is a frame we could work with
            found.append(curr_frame)
            found_times.append(curr_time)
                    
        all_found.append( (v, found) )
    
    print_flush("Candidate frames:")
    found_some = False
    for f in all_found:
        v,l = f
        print("{} : {}".format(v, l))
        if l:
            found_some = True
    
    if not found_some:
        print_flush("Found no interesting frames. Quitting...")
        import sys
        sys.exit(1)
        
    print_flush("Starting to train object detector with existing annotations...")
    
    input_shape = parse_resolution(input_shape)
    image_shape = parse_resolution(image_shape)
    
    model, bbox_util = train(dataset, import_datasets, input_shape, batch_size, epochs, frozen_layers, train_amount=1.0)
    
    print_flush("Applying the model to the images to find objects of type '{}'".format(class_name))
    
    masker = Masker(dataset)
    inputs = []
    frame_nums = []
    im_origs = []
    vids = []
    
    found_data = []
    
    for f in all_found:
        v,l = f

        with iio.get_reader(v) as vid:
            for frame_number in l:
                im_orig = vid.get_data(frame_number)
                im = im_orig.copy()
                im = masker.mask(im)
                
                resized = cv2.resize(im, (input_shape[0], input_shape[1]))
                inputs.append(resized)
                frame_nums.append(frame_number)
                im_origs.append(im_orig)
                vids.append(v)
        
                if len(inputs) == batch_size2:
                    tmp = process(inputs, frame_nums, im_origs, vids, confidence, 
                                  class_name, soft, batch_size2, model, 
                                  bbox_util, classes)
                                  
                    found_data.extend(tmp)
                                    
                    inputs = []
                    frame_nums = []
                    im_origs = []
                    vids = []
    
    if inputs:
        # There are still some leftovers
        tmp = process(inputs, frame_nums, im_origs, vids, confidence, 
                      class_name, soft, len(inputs), model, bbox_util, classes)
                      
        found_data.extend(tmp)
    
    print_flush("Writing images...")
    for x in found_data:
        v,f,im = x
        
        im_folder = datasets_path / dataset / "objects" / "train" / v.stem
        im_num = max([int(x.stem) for x in im_folder.glob('*.jpg')]) + 1
        im_path = im_folder / "{}.jpg".format(im_num)
        
        iio.imwrite(im_path, im)
        print_flush("Written {}".format(im_path))
        
        # Add the new frame numbers to frames.log for this video
        flog = im_folder / "frames.log"
        with flog.open('a') as log:
            log.write(str(f) + ' ')
    
    print_flush("Done!")
                
if __name__ == '__main__':
    rare_class_mining()


