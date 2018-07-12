"""script to train the ssd framework on PASCAL VOC."""

import click

from datetime import datetime
from glob import glob
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras import backend as K
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from scipy.misc import imread
import os

import sys
sys.path.append('../../apps/ssd_keras')

from ssd import SSD300
from create_prior_box import create_prior_box
from ssd_utils import BBoxUtility
from load_data import LoadDetections
import tensorflow as tf
import matplotlib as mpl
mpl.use('pdf')

from apply_mask import Masker
from util import parse_resolution

np.set_printoptions(suppress=True)

# some constants
BASE_LR = 3e-4


def get_width_height(filename):
    with Image.open(filename) as img:
        width, height = img.size
    return width, height


def log(string):
    print('[{time}] {string}'.format(time=datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), string=string))


def load_images(max_images, val_keys_file=None):
    if val_keys_file is not None:
        image_files = pickle.load(open(os.path.join('/deep_learning', val_keys_file), 'rb'))
    else:
        image_files = glob('/data/*.jpg') + glob('/data/*.png') + glob('/data/*.PNG')
    if max_images > 0:
        image_files = np.random.choice(image_files, max_images)

    df_image_files = pd.DataFrame(image_files, columns=['filename'])
    df_image_files[['width', 'height']] = df_image_files.filename.apply(get_width_height).apply(pd.Series)

    return df_image_files


def predict_bounding_boxes(df_image_files, batch_size, model, bbox_util, name, soft=False, sigma=0.5, input_shape=(300,300,3)):
    masker = Masker(name)
    results = []
    image_index = []
    for start_index in range(0, len(df_image_files), batch_size):
        end_index = min(start_index + batch_size, len(df_image_files))
        batch_image_files = df_image_files[start_index:end_index]
        inputs = []
        images = []
        indices = []

        print('Predicting block {} of {}'.format(start_index // batch_size + 1, int(np.ceil(len(df_image_files) / batch_size))))
        for index, batch_image_file in batch_image_files.iterrows():
            img = image.load_img(batch_image_file.filename, target_size=(input_shape[1], input_shape[0]))
            img = masker.mask(image.img_to_array(img)).astype(np.float32)
            images.append(imread(batch_image_file.filename))
            inputs.append(img.copy())
            indices.append(index)
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=4, verbose=1)
        print('Going to convert predictions into bbox_util')
        result = bbox_util.detection_out(preds, soft=soft, sigma=sigma)
        print('Done converting prediction into bbox util')
        result = [r if len(r) > 0 else np.zeros((1, 6)) for r in result]
        image_index += [[i] * len(r) for i, r in zip(indices, result)]
        results += result
    print('Going to convert to raw_detections')
    raw_detections = pd.DataFrame(np.vstack(results), columns=['class_index', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    raw_detections['image_index'] = np.hstack(image_index)
    return raw_detections


def convert_bounding_boxes(raw_detections, df_image_files, classes):
    detections = raw_detections
    detections.class_index = detections.class_index.astype(int)
    detections['class_name'] = detections.class_index.apply(lambda x: classes[x])
    detections['filename'] = df_image_files.filename[detections.image_index].values
    detections['width'] = df_image_files.width[detections.image_index].values
    detections['height'] = df_image_files.height[detections.image_index].values
    detections = detections.fillna(0)
    names = ['xmin', 'ymin', 'xmax', 'ymax']
    detections[names] = detections.apply(lambda x: x[names] * ([x.width, x.height] * 2), axis=1).astype('uint32')

    return detections


def plot_results(detections, num_classes, name):
    import matplotlib.pyplot as plt
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    for image_file, im_detections in detections.groupby('filename'):
        filename = os.path.join('/deep_learning', name, 'results', image_file.split(os.sep)[-1])
        aspect_ratio = im_detections.iloc[0].width / im_detections.iloc[0].height
        size = 10
        fig = plt.figure(figsize=(size, size/aspect_ratio))
        ax = fig.add_axes([0, 0, 1, 1])
        img = imread(image_file)
        plt.imshow(img / 255.)
        plt.axis('off')
        for _, det in im_detections.iterrows():
            score = det.confidence
            label_name = det.class_name
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (det.xmin, det.ymin), det.xmax-det.xmin+1, det.ymax-det.ymin+1
            color = colors[det.class_index]
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            ax.text(det.xmin, det.ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        plt.savefig(filename)
        plt.close(fig)


@click.command()
@click.option('--batch_size', default=32, help='The batch size of the training.')
@click.option('--max_images', default=-1, help='The maximum number of images used in the training.')
@click.option('--max_images_per_output', default=-1, help='The maximum number of images per output.')
@click.option('--name', prompt='The name of the trainset', help='The name of the trainset.')
@click.option('--threshold', default=0.6, help='The object threshold (0 < th < 1).')
@click.option('--val_keys_file', default=None, help='The pickled validation files list.')
@click.option('--save_images', is_flag=True, help='Save images with detections')
@click.option('--soft', is_flag=True, help='Use soft non maximum supression.')
@click.option('--sigma', default=0.5, help='The sigma for the soft non maximum suppression.')
@click.option('--input_shape',default='(300,300,3)',help='The size into which the images are rescaled before going into SSD')
@click.option('--experiment', default='default', help='The name of the experiment.')
@click.option('--train_data_dir', help='The location of the train data, when using matlab_export.')
@click.option('--memory_fraction', default=1.0, help='The memory fraction of the GPU memory to use for TensorFlow')
def main(batch_size, max_images, max_images_per_output, name, threshold,
         val_keys_file, save_images, soft, sigma, input_shape,experiment,train_data_dir,memory_fraction):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_session(tf.Session(config=config))
    
    input_shape = parse_resolution(input_shape)
    
    if val_keys_file == 'None':
        val_keys_file = None
    load_detections = LoadDetections()
    session = tf.Session()
    K.set_session(session)
    log('Started TensorFlow session')
    log('Chosen input_shape is {}'.format(input_shape))
    detections_file = '/deep_learning/{name}_{experiment}/detections.pickle'.format(name=name,experiment=experiment)
    os.makedirs('/deep_learning/{name}_{experiment}'.format(name=name,experiment=experiment), exist_ok=True)
    log('Loading detections')
    if os.path.exists(detections_file):
        detection_gt = pd.read_pickle(detections_file)
        log('Detections loaded from file')
    else:
        if name == 'matlab_export':
            train_data_dir = '/data/' + train_data_dir
            detection_gt = getattr(load_detections, name)(train_data_dir)
            log('Loading data from {}'.format(train_data_dir))
        else:
            detection_gt = getattr(load_detections, name)()

    log('Determining classes')
    classes = ['background'] + sorted(detection_gt.type.unique())

    print(' ')
    print('Classes:')
    print(classes)
    print(' ')
    num_classes = len(classes)


    os.makedirs('/deep_learning/{name}_{experiment}/results'.format(name=name,experiment=experiment), exist_ok=True)

    log('Loading model')
    model = SSD300((input_shape[1],input_shape[0],input_shape[2]), num_classes=num_classes)
    weights_files = glob(os.path.join('/deep_learning', '{name}_{experiment}'.format(name=name,experiment=experiment), 'checkpoints', '*.hdf5'))
    weights_files_loss = np.array([float(wf.split('-')[-1].replace('.hdf5', '')) for wf in weights_files])
    weights_file = weights_files[np.argmin(weights_files_loss)]
    model.load_weights(weights_file, by_name=True)
    log('Model loaded from {}'.format(weights_file))
    
    log('Generating priors')
    im_in = np.random.random((1,input_shape[1],input_shape[0],input_shape[2]))
    priors = model.predict(im_in,batch_size=1)[0, :, -8:]
    bbox_util = BBoxUtility(num_classes, priors)

    log('Loading images.')
    all_image_files = load_images(max_images, val_keys_file)
    save_file = '/deep_learning/{name}_{experiment}/results/df_image_files.csv'.format(name=name,experiment=experiment)
    all_image_files.to_csv(save_file)

    if max_images_per_output < 0:
        if max_images < 0:
            max_images_per_output = len(all_image_files)
        else:
            max_images_per_output = max_images

    for start_index in range(0, len(all_image_files), max_images_per_output):
        end_index = min(start_index + max_images_per_output, len(all_image_files))
        df_image_files = all_image_files[start_index:end_index].copy().reset_index(drop=True)
        log('Predicting image set {} of {}'.format(start_index // max_images_per_output + 1,
                                                   int(np.ceil(len(all_image_files) / max_images_per_output))))
        log('Predicting bounding boxes')
        raw_detections = predict_bounding_boxes(df_image_files, batch_size, model, bbox_util, name, soft, sigma, input_shape=input_shape)

        log('Converting bounding boxes')
        detections = convert_bounding_boxes(raw_detections, df_image_files, classes)

        log('Saving detections')
        save_file = '/deep_learning/{name}_{experiment}/results/detections_{index}.csv'.format(name=name,experiment=experiment, index=start_index)
        detections.to_csv(save_file)
        log('Detections saved to: {}'.format(save_file))

        if save_images:
            log('Plotting results.')
            plot_results(detections[detections.confidence > threshold], num_classes, '{name}_{experiment}'.format(name=name,experiment=experiment))

        log('Creating map csvs.')
        save_file = '/deep_learning/{name}_{experiment}/results/test_{index}.csv'.format(name=name, experiment=experiment,index=start_index)
        columns = ['filename', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
        detections.to_csv(save_file, columns=columns, header=False, index=False)

    session.close()
    log('Finished TensorFlow session')


if __name__ == '__main__':
    main()
