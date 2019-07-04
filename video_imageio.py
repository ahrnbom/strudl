"""A script for running an already trained SSD model on a video, saving the result as
   both a video file which can be inspected by humans, and also as a text file. 
   Only uses imageio for video input/output which is nice for when OpenCV 
   is built without video encoding support. """
   
import cv2
import imageio as io
import numpy as np
import pandas as pd
import click

from visualize import draw, class_colors

from ssd import SSD300
from create_prior_box import create_prior_box
from ssd_utils import BBoxUtility
from keras.applications.imagenet_utils import preprocess_input

from apply_mask import Masker
from classnames import get_classnames
from util import parse_resolution, print_flush
from folder import runs_path
      
def rescale(df, index, factor):
    """ Rescales a data frame row, as integers. Used since detections are stored on scale 0-1 """
    s = df[index]
    s2 = [int(factor*x) for x in s]
    df[index] = s2      

def get_model(name, experiment, input_shape, num_classes=6, verbose=True):
    """ Gets an SSD model, with trained weights
    
        Arguments:
        name        -- name of the dataset
        experiment  -- name of this training run
        input_shape -- size of images fed to SSD as a tuple like (640,480,3)
        num_classes -- the number of different object classes (including background)
    """
    model = SSD300((input_shape[1],input_shape[0],input_shape[2]), num_classes=num_classes)
    weights_files = list((runs_path / "{}_{}".format(name,experiment) / "checkpoints").glob('*.hdf5'))
    weights_files_loss = np.array([float(wf.stem.split('-')[-1]) for wf in weights_files])
    weights_file = weights_files[np.argmin(weights_files_loss)]
    model.load_weights(weights_file, by_name=True)
    if verbose:
        print_flush('Model loaded from {}'.format(weights_file))
    return model

def test_on_video(model, name, experiment, videopath, outvideopath, classnames, batch_size=32, input_shape=(480,640,3), soft=False,  width=480, height=640, conf_thresh=0.75, csv_conf_thresh=0.75):
    """ Applies a trained SSD model to a video
    
    Arguments:
    model           -- the SSD model, e.g. from get_model
    name            -- name of dataset
    experiment      -- name of training run
    videopath       -- path to input video
    outvideopath    -- path to output video showing the detections
    classnames      -- list of all the classes
    batch_size      -- number of images processed in parallell, lower this if you get out-of-memory errors
    input_shape     -- size of images fed to SSD
    soft            -- Whether to do soft NMS or normal NMS
    width           -- Width to scale detections with (can be set to 1 if detections are already on right scale)
    height          -- Height to scale detections with (can be set to 1 if detections are already on right scale)
    conf_thresh     -- Detections with confidences below this are not shown in output video. Set to negative to not visualize confidences.
    csv_conf_thresh -- Detections with confidences below this are ignored. This should be same as conf_thresh unless conf_thresh is negative.
    
    """
    masker = Masker(name)

    num_classes = len(classnames)+1
    colors = class_colors(num_classes)

    make_vid = True
    suffix = outvideopath.split('.')[-1]
    if suffix == 'csv':
        make_vid = False
        csvpath = outvideopath
    else:
        csvpath = outvideopath.replace('.{}'.format(suffix), '.csv')

    print_flush('Generating priors')
    im_in = np.random.random((1,input_shape[1],input_shape[0],input_shape[2]))
    priors = model.predict(im_in,batch_size=1)[0, :, -8:]
    bbox_util = BBoxUtility(num_classes, priors)    
        
    vid = io.get_reader(videopath)
    if make_vid: 
        outvid = io.get_writer(outvideopath, fps=30)
    
    inputs = []
    frames = []
    
    all_detections = []
    for i,frame in enumerate(vid):
        frame = masker.mask(frame)
        resized = cv2.resize(frame, (input_shape[0], input_shape[1]))
        
        frames.append(frame.copy())
        inputs.append(resized)
        
        if len(inputs) == batch_size:
            inputs = np.array(inputs).astype(np.float64)
            inputs = preprocess_input(inputs)
            
            preds = model.predict(inputs, batch_size=batch_size, verbose=0)
            results = bbox_util.detection_out(preds, soft=soft)
            
            for result, frame, frame_number in zip(results, frames, range(i-batch_size, i)):
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
                
                raw_detections['frame_number'] = (frame_number+2)
                all_detections.append(raw_detections[raw_detections.confidence>csv_conf_thresh])
                
                if make_vid: 
                    frame = draw(frame, raw_detections, colors, conf_thresh=conf_thresh)
                    outvid.append_data(frame)

            frames = []
            inputs = []

        if i%(10*batch_size) == 0:
            print_flush(i)
                    
    detections = pd.concat(all_detections)
    
    detections.to_csv(csvpath)

@click.command()
@click.option("--dataset", default="sweden2", help="Name of the dataset to use")
@click.option("--run", default="default", help="Name of the training run to use")
@click.option("--res", default="(640,480,3)", help="Image resolution for the object detector, as a string on the format '(640,480,3)' for example, with width, height and channels in that order.")
@click.option("--conf", default=0.6, type=float, help="Confidence threshold on detections")
@click.argument("invid", type=click.Path(exists=True))
@click.argument("outvid")
def main(dataset, run, res, conf, invid, outvid):
    res = parse_resolution(res)
    model = get_model(dataset, run, input_shape=(res[0], res[1], 3))
    classnames = get_classnames(dataset)
    test_on_video(model, dataset, run, invid, outvid, classnames, width=res[0], height=res[1], input_shape=(res[0], res[1], 3), conf_thresh=conf, csv_conf_thresh=conf)
    
if __name__ == "__main__":
    main()
