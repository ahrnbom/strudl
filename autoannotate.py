""" Module for training SSD on incomplete annotations, and using it to create
    automatic annotations for all the non-annotated images. Much of the code
    is copied from training_script, but creating a single script for both
    purposes was deemed cumbersome. Note that there is no early stopping, and the
    weight files are not stored anywhere. 
"""
    
import keras
from keras.callbacks import LearningRateScheduler
from keras.applications.imagenet_utils import preprocess_input
import click
import numpy as np
from random import shuffle
from os.path import isfile
from glob import glob
import imageio as iio
import pandas as pd
import cv2

from classnames import get_classnames
from ssd import SSD300
from create_prior_box import create_prior_box
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from load_data import LoadDetections

from apply_mask import Masker
from util import parse_resolution, print_flush, pandas_loop
from folder import runs_path, ssd_path, datasets_path

from training_script import Generator, schedule, get_image_props, detections_add_ytrue

# some constants
BASE_LR = 3e-4

def rep_last(some_list, n):
    """ Makes a list have a number of elements divisible by n, by repeating the last entry """
    while not (len(some_list)%n == 0):
        some_list.append(some_list[-1])
    
    return some_list

def get_images_to_autoannotate(dataset):
    train_path = "{dsp}{ds}/objects/train/".format(dsp=datasets_path, ds=dataset)
    images = glob("{tp}/*/*.jpg".format(tp=train_path))
    
    good_images = []
    for image in images:
        gtpath = image.replace('.jpg','.txt')
        if not isfile(gtpath):
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

    input_shape = parse_resolution(input_shape)
    image_shape = parse_resolution(image_shape)
    
    print_flush("Loading ground truth...")
    load_detections = LoadDetections()
    datasets = [dataset]
    if import_datasets:
        datasets.extend(import_datasets.split(','))

    detections = load_detections.custom(datasets)
    
    detections = detections.reset_index(drop=True)   
    image_props = get_image_props(detections)
    detections = detections_add_ytrue(detections, image_props, dataset)
    
    detections.index = detections.image_file
    
    print_flush('Ground truth object counts:')
    print_flush(detections.type.value_counts())
    
    classes = get_classnames(dataset)
    num_classes = len(classes) + 1
    
    keys = sorted(detections.image_file.unique())
    shuffle(keys)
    
    num_train = int(round(0.9 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]

    print_flush('Loading model...')
    model = SSD300((input_shape[1],input_shape[0],input_shape[2]), num_classes=num_classes)  
    model.load_weights(ssd_path+'weights_SSD300.hdf5', by_name=True)
    
    print_flush("Making priors...")    
    im_in = np.random.random((1,input_shape[1],input_shape[0],input_shape[2]))
    priors = model.predict(im_in,batch_size=1)[0, :, -8:]
    bbox_util = BBoxUtility(num_classes, priors)
    
    generator_kwargs = {
        'saturation_var': 0.5,
        'brightness_var': 0.5,
        'contrast_var': 0.5,
        'lighting_std': 0.5,
        'hflip_prob': 0.5,
        'vflip_prob': 0,
        'do_crop': True,
        'crop_area_range': [0.1, 1.0],
        'aspect_ratio_range': [0.5, 2]
        }

    path_prefix = ''
    gen = Generator(detections, bbox_util, batch_size, path_prefix,
                    train_keys, val_keys,
                    (input_shape[1], input_shape[0]), **generator_kwargs)

    # freeze several layers
    freeze = [
              ['input_1', 'conv1_1', 'conv1_2', 'pool1'],
              ['conv2_1', 'conv2_2', 'pool2'],
              ['conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
              ['conv4_1', 'conv4_2', 'conv4_3', 'pool4'],
              ['conv5_1', 'conv5_2', 'conv5_3', 'pool5'],
              ][:min(frozen_layers, 5)]

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False
    
    callbacks = [LearningRateScheduler(schedule)]
    
    optim = keras.optimizers.Adam(lr=BASE_LR / 10)
    model.compile(optimizer=optim, loss=MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss)
    
    print_flush("Training...")
    history = model.fit_generator(gen.generate(True), steps_per_epoch=gen.train_batches,
                                  epochs=epochs, verbose=2, callbacks=callbacks,
                                  validation_data=gen.generate(False), validation_steps=gen.val_batches, workers=1)
  
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
            
            preds = model.predict(inputs, batch_size=batch_size, verbose=0)
            results = bbox_util.detection_out(preds, soft=soft)
            
            for result, res_path in zip(results, impaths):
                result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
                raw_detections = pd.DataFrame(np.vstack(result), columns=['class_index', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
                
                auto_path = res_path.replace('.jpg','.auto')
                
                # Sort detections by confidence, keeping the top ones
                # This seems to be more robust than a hard-coded confidence threshold
                # Note that a confidence threshold can be chosen in the annotation web UI
                n = 128
                dets = [x for x in pandas_loop(raw_detections)]
                dets.sort(key=lambda x: 1.0-x['confidence'])
                if len(dets) > n:
                    dets = dets[:n]
                
                with open(auto_path, 'w') as f:
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
    


    
