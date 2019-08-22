""" Module for training SSD on incomplete annotations, and using it to create
    automatic annotations for all the non-annotated images. Much of the code
    is copied from training_script, but creating a single script for both
    purposes was deemed cumbersome. Note that there is no early stopping, and the
    weight files are not stored anywhere. 
"""
    
from keras.applications.imagenet_utils import preprocess_input
import click
import numpy as np
from pathlib import Path
import imageio as iio
import pandas as pd
import cv2

from classnames import get_classnames
from apply_mask import Masker
from util import parse_resolution, print_flush, pandas_loop, rep_last
from folder import datasets_path
from train import train



def get_images_to_autoannotate(dataset):
    train_path = datasets_path / dataset / "objects" / "train"
    images = list(train_path.glob('*/*.jpg'))
    
    good_images = []
    for image in images:
        gtpath = image.with_suffix('.txt')
        if not gtpath.is_file():
            good_images.append(image)
    
    good_images.sort()
    return good_images

@click.command()
@click.option("--dataset", default="default", help="Name of dataset")
@click.option("--import_datasets", default="", type=str, help="Additional datasets to include during training, separated by commas")
@click.option('--input_shape',default='(300,300,3)',help='The size into which the images are rescaled before going into SSD')
@click.option('--image_shape', default='(300,300,3)',help='The size of the original training images')
@click.option('--batch_size', default=4, help='The batch size of the training.')
@click.option('--batch_size2', default=32, help='The batch size of the testing.')
@click.option('--epochs', default=75, help='The number of epochs used to run the training.')
@click.option('--frozen_layers', default=3, help='The number of frozen layers, max 5.')
def autoannotate(dataset, import_datasets, input_shape, image_shape, batch_size, batch_size2, epochs, frozen_layers):
    
    soft = False
    
    classes = get_classnames(dataset)
    
    input_shape = parse_resolution(input_shape)
    image_shape = parse_resolution(image_shape)
    
    model, bbox_util = train(dataset, import_datasets, input_shape, batch_size, epochs, frozen_layers)
    
    print_flush("Auto-annotating...")
    masker = Masker(dataset)
    
    inputs = []
    impaths = []
    to_annotate = get_images_to_autoannotate(dataset)
    
    # rep_last needed since we use large batches, for speed, to make sure we run on all images
    for impath in rep_last(to_annotate, batch_size2):
        im = iio.imread(impath)
        im = masker.mask(im)
        resized = cv2.resize(im, (input_shape[0], input_shape[1]))
        inputs.append(resized)
        impaths.append(impath)
        
        if len(inputs) == batch_size2:
            inputs = np.array(inputs).astype(np.float64)
            inputs = preprocess_input(inputs)
            
            preds = model.predict(inputs, batch_size=batch_size2, verbose=0)
            results = bbox_util.detection_out(preds, soft=soft)
            
            for result, res_path in zip(results, impaths):
                result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
                raw_detections = pd.DataFrame(np.vstack(result), columns=['class_index', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
                
                auto_path = res_path.with_suffix('.auto')
                
                # Sort detections by confidence, keeping the top ones
                # This seems to be more robust than a hard-coded confidence threshold
                # Note that a confidence threshold can be chosen in the annotation web UI
                n = 128
                dets = [x for x in pandas_loop(raw_detections)]
                dets.sort(key=lambda x: 1.0-x['confidence'])
                if len(dets) > n:
                    dets = dets[:n]
                
                with auto_path.open('w') as f:
                    for det in dets:
                        conf = round(det['confidence'],4)
                        line = "{index} {cx} {cy} {w} {h} conf:{conf} {cn}\n".format(index=int(det['class_index']),
                                     cx = round((det['xmin']+det['xmax'])/2,4),
                                     cy = round((det['ymin']+det['ymax'])/2,4),
                                     w = round(det['xmax']-det['xmin'],4),
                                     h = round(det['ymax']-det['ymin'],4),
                                     conf=conf,
                                     cn = classes[int(det['class_index'])-1])
                        f.write(line)
                print_flush("Wrote {}".format(auto_path))
                
            inputs = []
            impaths = []
            
    assert(not inputs) # If this fails, not all images were processed!
    print_flush("Done!")
            
if __name__ == '__main__':
    autoannotate()
    


    
