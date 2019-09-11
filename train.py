import keras
from keras.callbacks import LearningRateScheduler
from keras.applications.imagenet_utils import preprocess_input
import click
import numpy as np
from random import shuffle
from os.path import isfile
from pathlib import Path
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

def train(dataset, import_datasets, input_shape, batch_size, epochs, frozen_layers, train_amount=0.9):
    
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
    
    if train_amount < 1.0:
        num_train = int(round(train_amount * len(keys)))
        train_keys = keys[:num_train]
        val_keys = keys[num_train:]
    else:
        train_keys = keys
        
        # Not a very good validation set, but whatever.
        # The ability to train on all the images is important when annotations are sparse, 
        # like when doing autoannotation
        val_keys = [keys[0]] 

    print_flush('Loading model...')
    model = SSD300((input_shape[1],input_shape[0],input_shape[2]), num_classes=num_classes)  
    model.load_weights(ssd_path / 'weights_SSD300.hdf5', by_name=True)
    
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
    
    return model, bbox_util
  
