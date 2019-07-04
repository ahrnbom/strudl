"""script to train the ssd framework on PASCAL VOC."""

import shutil

import keras
import click
import random

from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import os

from ssd import SSD300
from create_prior_box import create_prior_box
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from load_data import LoadDetections

import tensorflow as tf
import logging
import subprocess

from apply_mask import Masker
from util import parse_resolution, print_flush, pandas_loop
from folder import runs_path, base_path, ssd_path, mkdir
from ssd_cleanup import cleanup
from classnames import get_classnames
from plot import multi_plot

np.set_printoptions(suppress=True)

# some constants
BASE_LR = 3e-4

class Generator(object):
    """ Generator to generate training data """

    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 crop_attempts=10,
                 aspect_ratio_range=[3. / 4., 4. / 3.]):

        # Because we can include multiple datasets, we need to keep track of multiple masks. They are stored here, as a dict from dataset names to the maskers.
        self.maskers = dict()

        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = int(np.ceil(len(train_keys) / batch_size))
        self.val_batches = int(np.ceil(len(val_keys) / batch_size))
        self.image_size = image_size
        self.crop_attempts = crop_attempts
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_w)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True, do_shuffle=True):
        inputs = []
        targets = []
        while True:
            if train:
                if do_shuffle:
                    shuffle(self.train_keys)
                keys = self.train_keys
            else:
                if do_shuffle:
                    shuffle(self.val_keys)
                keys = self.val_keys
            for key in keys:
                img_path = self.path_prefix + key

                # To know the correct mask, we need the dataset name
                dataset = img_path.split('/')[3]
                if dataset in self.maskers:
                    masker = self.maskers[dataset]
                else:
                    masker = Masker(dataset)
                    self.maskers[dataset] = masker

                img = masker.mask(imread(img_path)).astype('float32')

                y = np.vstack(self.gt.loc[key].y_true)
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets


def schedule(epoch, decay=0.9):
    return BASE_LR * decay**(epoch)


def get_width_height(filename):
    with Image.open(filename) as img:
        width, height = img.size
    return width, height


def get_image_props(detections):
    image_props = {image_file: get_width_height(image_file) for image_file in detections.image_file.unique()}
    return image_props


def detections_add_ytrue(detections, image_props, dataset):
    types = get_classnames(dataset) #sorted(detections.type.unique())
    y_true = pd.Series()
    for image_file, det in detections.groupby('image_file'):
        xymin_xymax = np.array([det.xmin.values, det.ymin.values, det.xmax.values, det.ymax.values]).T
        classes = np.array([[d.type == t for t in types] for _, d in det.iterrows()], dtype=np.uint8)
        y_true = y_true.append(pd.Series(np.concatenate([xymin_xymax, classes], axis=1).tolist(), index=det.index))
    y_true = y_true.apply(lambda x: np.array(x)[np.newaxis, :])
    detections['y_true'] = y_true
    return detections


def log(string):
    outstring = '[{time}] {string}'.format(time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), string=string)
    logging.info(outstring)
    print(outstring)


@click.command()
@click.option('--batch_size', default=32, help='The batch size of the training.')
@click.option('--max_images', default=-1, help='The maximum number of images used in the training.')
@click.option('--epochs', default=100, help='The number of epochs used to run the training.')
@click.option('--frozen_layers', default=3, help='The number of frozen layers, max 5.')
@click.option('--experiment', default='default', help='The name of the experiment.')
@click.option('--name', prompt='The name of the trainset', help='The name of the trainset.')
@click.option("--import_datasets", default="", type=str, help="Additional datasets to include during training, separated by commas")
@click.option('--train_data_dir', prompt='The location of the train data, when using matlab_export', help='The location of the train data, when using matlab_export.')
@click.option('--input_shape',default='(300,300,3)',help='The size into which the images are rescaled before going into SSD')
@click.option('--image_shape', default='(300,300,3)',help='The size of the original training images')
@click.option('--memory_fraction', default=0.95, help='The memory fraction of the GPU memory to use for TensorFlow')
@click.option('--do_crop', is_flag=True, help='Use random crops of images during training.')
def main(batch_size, max_images, epochs, name, import_datasets, frozen_layers, experiment,train_data_dir,input_shape,image_shape,memory_fraction,do_crop):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_session(tf.Session(config=config))
    
    run_name = "{}_{}".format(name,experiment)
    
    input_shape = parse_resolution(input_shape)
    image_shape = parse_resolution(image_shape)

    load_detections = LoadDetections()
    session = tf.Session()
    K.set_session(session)
    log('Started TensorFlow session')
    log('Chosen input_shape is {}'.format(input_shape))
    detections_file = runs_path / run_name / "detections.pickle"
    mkdir(runs_path / run_name)
    
    logging.basicConfig(filename=str(runs_path / run_name / "trainlog.log"), level=logging.INFO)

    try:
        githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()[0:6].decode('utf-8')
        log("Git hash: {}".format(githash))
    except subprocess.CalledProcessError:
        pass

    log('Loading detections')

    datasets = [name]
    if import_datasets:
        datasets.extend(import_datasets.split(','))
        log('Using these datasets: ' + str(datasets))

    detections = load_detections.custom(datasets)

    log('Detections loaded')
    log('Calculating image properties')
    detections = detections.reset_index(drop=True)
    image_props = get_image_props(detections)
    log('Image properties created')

    log('Adding y_true to detections')
    detections = detections_add_ytrue(detections, image_props, name)

    detections.index = detections.image_file
    print(' ')
    print('Detection frequencies:')
    print(detections.type.value_counts())
    print(' ')
    classes = get_classnames(name) #sorted(detections.type.unique())
    num_classes = len(classes) + 1

    log('Loading priors')

    keys = sorted(detections.image_file.unique())
    random.shuffle(keys)
    if max_images > 0:
        keys = keys[:max_images]
    shuffle(keys)
    num_train = int(round(0.9 * len(keys)))
    if num_train == len(keys):
        num_train -= 1
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    train_keys_file = runs_path / run_name / "train_keys.pickle"
    log('Saving training keys to: {}'.format(train_keys_file))
    pickle.dump(str(train_keys), train_keys_file.open('wb'))
    val_keys_file = runs_path / run_name / "val_keys.pickle"
    log('Saving validation keys to: {}'.format(val_keys_file))
    pickle.dump(str(val_keys), val_keys_file.open('wb'))

    log('Loading model')
    model = SSD300((input_shape[1],input_shape[0],input_shape[2]), num_classes=num_classes)
    model.load_weights(ssd_path / "weights_SSD300.hdf5", by_name=True)

    log('Generating priors')
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
        'do_crop': do_crop,
        'crop_area_range': [0.1, 1.0],
        'aspect_ratio_range': [0.5, 2]
        }

    path_prefix = ''
    gen = Generator(detections, bbox_util, batch_size, path_prefix,
                    train_keys, val_keys,
                    (input_shape[1], input_shape[0]), **generator_kwargs)

    # freeze several layers
    # freeze = []
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
    mkdir(runs_path / run_name / "checkpoints")
    shutil.rmtree( str(runs_path / run_name / "logs"), ignore_errors=True )
    mkdir(runs_path / run_name / "logs")

    callbacks = [ModelCheckpoint(str(runs_path / run_name / 'checkpoints') + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                 verbose=2, save_weights_only=True),
                 TensorBoard(log_dir=str(runs_path / run_name / "logs"), write_graph=False),
                 LearningRateScheduler(schedule)]

    optim = keras.optimizers.Adam(lr=BASE_LR / 10)
    # optim = keras.optimizers.RMSprop(lr=BASE_LR / 10)
    model.compile(optimizer=optim, loss=MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss)

    log('Running model')
    history = model.fit_generator(gen.generate(True), steps_per_epoch=gen.train_batches,
                                  epochs=epochs, verbose=2, callbacks=callbacks,
                                  validation_data=gen.generate(False), validation_steps=gen.val_batches, workers=1)
    log('Done training model')
    session.close()
    log('Session closed, starting with writing results')
    results = pd.DataFrame(history.history).unstack().reset_index(0)
    results = results.rename(columns={'level_0': 'type', 0: 'value'})
    
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for row in pandas_loop(results):
        if row['type'] == 'loss':
            x1.append(row['_'])
            y1.append(row['value'])
        elif row['type'] == 'val_loss':
            x2.append(row['_'])
            y2.append(row['value'])
    
    plot_path = runs_path / run_name / "training.png"
    multi_plot([x1,x2], [y1,y2], plot_path, xlabel='epochs', ylabel='loss', title='Training', legend=['loss', 'validation loss'])
    
    results.to_csv(runs_path / run_name / "results.csv")    

    log('Cleaning up non-optimal weights...')
    cleanup(name, experiment)

    log('Finished TensorFlow session')
    print_flush('Done!')


if __name__ == '__main__':
    main()
