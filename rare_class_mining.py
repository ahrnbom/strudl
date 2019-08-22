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

from folder import datasets_path
from timestamps import Timestamps
from util import parse_resolution, print_flush, pandas_loop, rep_last
from apply_mask import Masker
from classnames import get_classnames
from train import train

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
def rare_class_mining(dataset, class_name, time_dist, sampling_rate, import_datasets, input_shape, image_shape, batch_size, batch_size2, epochs, frozen_layers):
    soft = False
    classes = get_classnames(dataset)
    
    ts = Timestamps(dataset)

    # Find all videos in dataset
    vidnames = list((datasets_path / dataset / "videos").glob('*.mkv'))
    
    all_found = []
    
    for v in vidnames:
        # Find existing annotations
        frames_log = (datasets_path / dataset / "objects" / "train" / v.stem / "frames.log").read_text().split('\n')
        frames_log = [x for x in frames_log[1:] if x] # Remove first line, which is video name, and any empty lines
        annotated = [int(x) for x in frames_log]
        
        curr_time = ts.get(v.stem, 0)
        prev_frame = 0
        
        annotated_times = [ts.get(v.stem, x) for x in annotated]
        
        found = []
        done = False
        
        while not done:
            # Sample in time
            curr_time += timedelta(seconds=sampling_rate)            
            curr_frame = ts.get_frame_number_given_vidname(curr_time, v.stem)

            if curr_frame == prev_frame:
                # We have reached the end of the video
                done = True
                continue
                
            prev_frame = curr_frame
            
            # Check if we are too close to existing annotations
            dists = [abs((curr_time-x).total_seconds()) for x in annotated_times]
            if any([(x <= time_dist) for x in dists]):
                continue
            
            # This is a frame we could work with
            found.append(curr_frame)
        
        all_found.append( (v.stem, found) )
    
    print_flush("Found the following frames worth checking out:")
    for f in all_found:
        v,l = f
        print_flush(v + ":")
        print_flush(",".join([str(x) for x in l]))
        print_flush("")
    
    print_flush("Starting to train object detector with existing annotations...")
    
    input_shape = parse_resolution(input_shape)
    image_shape = parse_resolution(image_shape)
    
    model = train(dataset, import_datasets, input_shape, batch_size, epochs, frozen_layers)
    
    print_flush("Applying the model to the images to find objects of type", class_name)
    
    masker = Masker(dataset)
    inputs = []
    
    found_frames = []
    
    # rep_last needed since we use large batches, for speed, to make sure we run on all images
    for f in all_found:
        v,l = f
        with iio.get_reader(v) as vid:
            for frame_number in l:
                im_orig = vid.get_data(frame_number)
                im = im_orig.copy()
                im = masker.mask(im)
                
                resized = cv2.resize(im, (input_shape[0], input_shape[1]))
                inputs.append(resized)
        
                if len(inputs) == batch_size2:
                    inputs = np.array(inputs).astype(np.float64)
                    inputs = preprocess_input(inputs)
                    
                    preds = model.predict(inputs, batch_size=batch_size2, verbose=0)
                    results = bbox_util.detection_out(preds, soft=soft)
                    
                    for result, res_path in zip(results, impaths):
                        result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
                        for r in result:
                            this_class_name = classes[r[0]]
                            if this_class_name == class_name:
                                found_frames.append( (v.stem, frame_number, im_orig) )
                                print_flush("Found an object of class",class_name,"in frame", frame_number, "in video", v.stem)
    
    print_flush("Writing images...")
    for x in found_frames:
        v,f,im = x
        
        im_folder = datasets_path / dataset / "objects" / "train" / v.stem 
        im_num = max([int(x.stem) for x in im_folder.glob('*.jpg')]) + 1
        im_path = im_folder / "{}.jpg".format(im_num)
        
        iio.imwrite(im_path, im)
        print_flush("Written",im_path)
                
if __name__ == '__main__':
    rare_class_mining()


