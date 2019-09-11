""" The module for starting a web server, presenting the Web UI and the API defined
    by strudl.yaml. It should be the main entrypoint for running strudl (like `python server.py`)
    Each function here is mapped to from strudl.yaml, to be the response to API calls.
    The documentation for how these functions work is in strudl.yaml so it's not repeated here.
    The main function at the bottom is responsible for starting the server using connexion.
"""

import connexion
from connexion import NoContent
from shlex import quote
from flask import send_from_directory, send_file
from glob import glob
from pathlib import Path
import os
import cv2
from random import choice
import subprocess
import click
import sys

from jobman import JobManager
from config import DatasetConfig, RunConfig
from folder import mkdir, datasets_path, runs_path, ssd_path
from classnames import get_classnames, set_class_data
from tracking import DetTrack # not directly used, but required for tracks_formats to work for some reason
from tracks_formats import format_tracks, generate_tracks_in_zip, all_track_formats
from import_videos import import_videos
from visualize import class_colors
from visualize_objects import slideshow
from validation import validate_annotation, validate_calibration, validate_pretrained_md5
from annotation import annotation_image_list, get_annotation_path, get_annotation_object, annotation_data
from storage import load, save
from tracking_world import WorldTrackingConfig, WorldTrack # same as DetTrack
from compstatus import status
from util import left_remove, right_remove

jm = JobManager()
python_path = sys.executable

def get_progress(dataset_name, run_name):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    
    ds_path = datasets_path / dataset_name
    
    if ds_path.is_dir():
        progress = dict()
        
        progress['has_config'] = (ds_path / 'config.txt').is_file()
        if progress['has_config']: 
            dc = DatasetConfig(dataset_name)
            
        progress['has_mask'] = (ds_path / 'mask.png').is_file()
        progress['has_classnames'] = (ds_path / 'classes.txt').is_file()
        progress['has_calibration'] = (ds_path / 'calib.tacal').is_file()
        
        progress['number_of_timestamp_logs'] = len(list( (ds_path / "logs").glob('*.log') ))
        progress['number_of_videos'] = len(list((ds_path / 'videos').glob('*.mkv')))
        progress['training_frames_to_annotate'] = len(list( (ds_path / "objects" / "train").glob('*/*.jpg') ))
        progress['training_frames_annotated'] = len(list( (ds_path / "objects" / "train").glob('*/*.txt') ))
        
        progress['videos_with_point_tracks_computed'] = len(list( (ds_path / "klt").glob('*.pklz') ))
        progress['videos_with_point_tracks_visualized'] = len(list( (ds_path / "klt").glob('*.mp4') ))
        
        progress['all_runs'] = [x.stem.split('_')[-1] for x in runs_path.glob(dataset_name + '_*')]
        
        run_path = runs_path / (dataset_name + '_' + run_name)
        if run_path.is_dir():
            progress['has_this_run'] = True
            
            rprogress = dict()
            rprogress['has_pretrained_weights'] = (ssd_path / 'weights_SSD300.hdf5').is_file()
            rprogress['videos_with_detected_objects'] = len(list( run_path.glob('csv/*.csv') ))
            rprogress['videos_with_detected_objects_visualized'] = len(list(run_path.glob('detections/*.mp4')))
            rprogress['videos_with_detected_objects_in_world_coordinates'] = len(list(run_path.glob('detections_world/*.csv')))
            rprogress['videos_with_detected_objects_in_world_coordinates_visualized'] = len(list(run_path.glob('detections_world/*.mp4')))
            rprogress['stored_weight_files'] = len(list(run_path.glob('checkpoints/*.hdf5')))
            rprogress['videos_with_pixel_coordinate_tracks'] = len(list(run_path.glob('tracks/*.pklz')))
            rprogress['videos_with_pixel_coordinate_tracks_visualized'] = len(list(run_path.glob('tracks/*.mp4')))
            rprogress['videos_with_world_coordinate_tracks'] = len(list(run_path.glob('tracks_world/*.pklz')))
            rprogress['videos_with_world_coordinate_tracks_visualized'] = len(list(run_path.glob('tracks_world/*.mp4')))
            rprogress['has_optimized_world_tracking'] = (run_path / 'world_tracking_optimization.pklz').is_file()
            rprogress['has_visualized_optimized_world_tracking'] = (run_path / 'world_tracking_optimization.mp4').is_file()
            rprogress['has_world_tracking_ground_truth'] = (run_path / 'world_trajectory_gt.csv').is_file()
            rprogress['track_zips'] = [x.name for x in run_path.glob('track_zips/*.zip')]
            
            all_progress = {'dataset': progress, 'run': rprogress}
        else:
            progress['has_this_run'] = False    
            all_progress = {'dataset': progress}
        
        return (all_progress, 200)
    else:
        return ("Dataset does not exist", 404)
    

def give_access_to_data():
    # This feels like a security hazard, but at least the path is hardcoded

    cmd = ['chmod', '-R', '777', '/data']
    completed = subprocess.run(cmd)
    if completed.returncode == 0:
        return (NoContent, 200)
    else:
        return (completed.returncode, 500)

def annotation_page():
    return send_from_directory('webui', 'annot.html')

def index_page():
    return send_from_directory('webui', 'index.html')

def get_list_of_annotation_images(dataset_name, annotation_set):
    dataset_name = quote(dataset_name)
    annotation_set = quote(annotation_set)
    
    out = annotation_image_list(dataset_name, annotation_set)
    
    if out is None:
        return (NoContent, 404)
    else:  
        return (out, 200)        

def get_annotation_annotation(dataset_name, image_number, video_name, annotation_set, output_format, accept_auto=False):
    suffixes = ['.txt']
    if accept_auto:
        suffixes.append('.auto')

    if output_format == 'plain':
        return get_annotation(dataset_name, image_number, video_name, annotation_set, suffixes, 'text/plain')
    elif output_format == 'json':
        impath = get_annotation(dataset_name, image_number, video_name, annotation_set, suffixes, 'text/plain', send=False)
        
        if not (impath is None):
            annots = get_annotation_object(impath)
            return (annots, 200)
                
        else:
            return (NoContent, 404)
    else:
        return (NoContent, 400)
        
def get_annotation_image(dataset_name, image_number, video_name, annotation_set):
    return get_annotation(dataset_name, image_number, video_name, annotation_set, '.jpg', 'image/jpeg')

def get_annotation(dataset_name, image_number, video_name, annotation_set, suffix, mime, send=True):
    dataset_name, video_name, annotation_set = map(quote, (dataset_name, video_name, annotation_set))
    
    if not (type(suffix) == list):
        suffix = [suffix]
    
    impath = None
    for sfx in suffix:
        impath = get_annotation_path(dataset_name, annotation_set, video_name=video_name, image_number=image_number, suffix=sfx)
        if not (impath is None):
            break
    
    if send:   
        if impath is None:
            return (NoContent, 404)
        else:
            return send_file(str(impath), mimetype=mime)
    else:
        return impath

def get_annotation_slideshow(dataset_name):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        imsize = dc.get('video_resolution')
        outpath = datasets_path / dataset_name / "slideshow.mp4"
        res = slideshow(dataset_name, outpath)
        
        if not res:
            return ("Failed to make slideshow", 404)
        else:
            vid = send_file(str(outpath), mimetype='video/mp4')
            return (vid, 200)
    else:
        return ("Dataset does not exist", 404)
    
def post_annotation_annotation(dataset_name, image_number, video_name, annotation_set, annotation_text):
    dataset_name, video_name, annotation_set = map(quote, (dataset_name, video_name, annotation_set))

    annotation_text = annotation_text.decode('utf-8')
    if validate_annotation(annotation_text, dataset_name):   
        folder_path = datasets_path / dataset_name / "objects" / annotation_set / video_name
        if folder_path.is_dir():
            file_path = folder_path / "{}.txt".format(image_number)
        
            with file_path.open('w') as f:
                f.write(annotation_text)
        
            return (NoContent, 200)
        
        else:
            return (NoContent, 404)
    else:
        return (NoContent, 400)

def get_annotation_data(dataset_name):
    
    dataset_name = quote(dataset_name)
    out = annotation_data(dataset_name)
    
    if out is None:
        return (NoContent, 404)
    else:
        return (out, 200)
            
def post_dataset(dataset_name, class_names, class_heights):
    if ' ' in dataset_name:
        return ("Spaces are not allowed in dataset names!", 500)
        
    dataset_name = quote(dataset_name)
    path = datasets_path / dataset_name
    mkdir(path)
    mkdir(path / 'videos')

    class_names = [quote(x.lower()) for x in class_names.split(',')]
    class_heights = map(float, class_heights.split(','))
    class_data = [{'name': n, 'height': h} for n, h in zip(class_names, class_heights)]
    set_class_data(dataset_name, class_data)

    return (NoContent, 200)
    
def get_datasets():
    datasets = [x.name for x in datasets_path.glob('*') if x.is_dir()]
    datasets.sort()
    return (datasets, 200)

def get_dataset_config(dataset_name):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    if dc.exists:
        return dc.get_data()
    else:
        return (NoContent, 404)

def post_dataset_config(dataset_name, dataset_config):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    if dc.set_data(dataset_config):
        
        dc.save()
        return (NoContent, 200)
    else:
        return ("Could not interpret dataset configuration. Is some required parameter missing? Is video resolution divisible by 16?", 500)
    
def get_run_config(dataset_name, run_name):
    dataset_name = quote(dataset_name)
    rc = RunConfig(dataset_name, run_name)
    if rc.exists:
        return rc.get_data()
    else:
        return (NoContent, 404)
    
def post_run_config(dataset_name, run_name, run_config):
    if ' ' in run_name:
        return ("Spaces are not allowed in run names!", 500)

    if '_' in run_name:
        return ("Underscores are not allowed in run names!", 500)

    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    if rc.set_data(run_config):
        rc.save()
        return (NoContent, 200)
    else:
        return ("Could not interpret run configuration. Is some required parameter missing?", 500)

def get_pretrained_weights():
    path = ssd_path / 'weights_SSD300.hdf5'
    if path.is_file():
        return ("Already present", 200)
    else:
        url = 'https://github.com/hakanardo/weights/raw/d2243707493e2e5f94c465b6248558ee16c90be6/weights_SSD300.hdf5'
        mkdir(ssd_path)
        os.system("wget -O %s '%s'" % (path, url))
    if not path.is_file():
        return ("Download failed", 500)
    if validate_pretrained_md5(path):
        return ("Downloaded", 200)
    else:
        path.unlink()
        return ("File rejected", 500)

def post_pretrained_weights(weights_file):
    path = ssd_path / 'weights_SSD300.hdf5'
    weights_file.save(str(path))
    if validate_pretrained_md5(path):
        return (NoContent, 200)
    else:
        path.unlink()
        return ("File rejected", 400)
    
def post_mask(dataset_name, mask_image_file):
    dataset_name = quote(dataset_name)
    mask_tmp_path = datasets_path / dataset_name / "mask_tmp.png"
    mask_path = datasets_path / dataset_name / "mask.png"
    mask_image_file.save(str(mask_tmp_path))
    
    success = False
    try:
        # This is not really safe, but at least should protect from some completely broken image files
        im = cv2.imread(str(mask_tmp_path), -1)
        assert(im.shape[2] == 4)
        cv2.imwrite(str(mask_path), im)
        success = True
    except:
        success = False
        
    mask_tmp_path.unlink()
    
    if success:
        return (NoContent, 200)
    else:
        try:
            mask_path.unlink()
        except:
            pass
            
        return (NoContent, 500)
    
def get_mask(dataset_name):
    dataset_name = quote(dataset_name)
    mask_path = datasets_path / dataset_name / "mask.png"
    if mask_path.is_file():
        mask_file = send_file(str(mask_path), mimetype='image/png')
        return (mask_file, 200)
    else:
        return (NoContent, 404)

def get_job_status():
    running = jm.get_jobs("running")
    recent = jm.get_jobs("recent")
    recent_log = None
    
    if recent:
        recent = recent[-1]
        recent_log = jm.get_log(recent)
    
    obj = dict()
    obj['running_now'] = False
    if running:
        obj['running_now'] = True
    
    obj['latest_log'] = False
    if recent_log:
        obj['latest_log'] = recent_log.split('\n')
    
    cstatus = status()
    obj['cpu'], obj['ram'], obj['gpu'], obj['vram'], obj['disk'] = cstatus
    
    return (obj, 200)  
    
def get_job_ids(jobs_type):
    ids = jm.get_jobs(jobs_type)
    return (ids, 200)

def get_job_by_id(job_id):
    if job_id == "running":
        job_id = jm.get_jobs("running")
    elif job_id == "last":
        job_ids = jm.get_jobs("recent")

        if job_ids:
            job_id = job_ids[-1]
        else:
            return (NoContent, 404)
            
    log = jm.get_log(job_id)
        
    if log is None:
        return (NoContent, 404)
    else:
        return (log, 200)
        
def delete_job_by_id(job_id):
    res = jm.stop(job_id)
    if res:
        return (NoContent, 200)
    else:
        return (NoContent, 404)
    
    
def post_import_videos_job(dataset_name, path, method, logs_path=None, minutes=0):
    # The paths in this function are just strings, not Path objects
    
    dataset_name = quote(dataset_name)

    if logs_path is None:
        logs_path = path
        # Since 'path' probably contains a query, like ending with '*.mkv', this should be removed
        if not (logs_path[-1] == '/'):
            logs_path = right_remove(logs_path, logs_path.split('/')[-1])

    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        resolution = dc.get('video_resolution')
        fps = dc.get('video_fps')
        cmd = [python_path, "import_videos.py",
               "--query={}".format(path),
               "--dataset={}".format(dataset_name),
               "--resolution={}".format(resolution),
               "--method={}".format(method),
               "--fps={}".format(fps),
               "--logs={}".format(logs_path),
               "--minutes={}".format(minutes)]
       
        job_id = jm.run(cmd, "import_videos")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)       
               
    else:
        return (NoContent, 404)

def post_point_tracks_job(dataset_name, visualize, overwrite):
    assert(type(visualize) == bool)
    assert(type(overwrite) == bool)
    
    cmd = "findvids"
    if not overwrite:
        cmd = "continue"
    
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    if dc.exists:
        cmd = [python_path, "klt.py",
               "--cmd={}".format(cmd),
               "--dataset={}".format(dataset_name),
               "--imsize={}".format(dc.get('point_track_resolution')),
               "--visualize={}".format(visualize)]
        
        job_id = jm.run(cmd, "point_tracks")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
       
    else:
        return (NoContent, 404)
        
def post_prepare_annotations_job(dataset_name, less_night=True):
    assert(type(less_night) == bool)
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        cmd = [python_path, "annotation_preparation.py",
               "--dataset={}".format(dataset_name),
               "--num_ims={}".format(dc.get('images_to_annotate')),
               "--ims_per_vid={}".format(dc.get('images_to_annotate_per_video')),
               "--train_amount={}".format(dc.get('annotation_train_split')),
               "--night={}".format(less_night)]
        
        job_id = jm.run(cmd, "prepare_annotations")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)

def post_prepare_extra_annotations_job(dataset_name, times, images_per_time, interval_length=2.0):
    dataset_name = quote(dataset_name)
    times = quote(times)
    assert(type(images_per_time) == int)
    assert(type(interval_length) == float)
    
    cmd = [python_path, "extra_annotations.py",
           "--dataset={}".format(dataset_name),
           "--times={}".format(times),
           "--images_per_time={}".format(images_per_time),
           "--interval={}".format(interval_length)]
    
    job_id = jm.run(cmd, "extra_annotations")
    if job_id:
        return (job_id, 202)
    else:
        return (NoContent, 503)
    
        
def post_autoannotate_job(dataset_name, import_datasets="", epochs=75, resolution="(640,480,3)"):
    dataset_name = quote(dataset_name)
    resolution = quote(resolution)
    
    dc = DatasetConfig(dataset_name)
    if dc.exists:
        cmd = [python_path, "autoannotate.py",
               "--dataset={}".format(dataset_name),
               "--input_shape={}".format(resolution),
               "--image_shape={}".format(dc.get('video_resolution')),
               "--epochs={}".format(epochs)]
        
        if import_datasets:
            import_datasets = quote(import_datasets)
            cmd.append("--import_datasets={}".format(import_datasets))
            
        job_id = jm.run(cmd, "autoannotate")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)

def post_rare_class_mining_job(dataset_name, class_name, confidence, time_distance, 
                               time_sampling, import_datasets="", epochs=75, 
                               resolution="(300,300,3)"):
    dataset_name = quote(dataset_name)
    class_name = quote(class_name)
    resolution = quote(resolution)
    
    dc = DatasetConfig(dataset_name)
    if dc.exists:
        cmd = [python_path, "rare_class_mining.py",
               "--dataset={}".format(dataset_name),
               "--class_name={}".format(class_name),
               "--confidence={}".format(confidence),
               "--time_dist={}".format(time_distance),
               "--sampling_rate={}".format(time_sampling),
               "--epochs={}".format(epochs),
               "--input_shape={}".format(resolution),
               "--image_shape={}".format(dc.get('video_resolution'))]
        
        if import_datasets:
            import_datasets = quote(import_datasets)
            cmd.append("--import_datasets={}".format(import_datasets))
        
        job_id = jm.run(cmd, "rare_class_mining")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return ("Dataset does not exists or is not configured", 404)
    
def post_train_detector_job(dataset_name, run_name, epochs, import_datasets=""):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = [python_path, "training_script.py", 
        "--name={}".format(dataset_name), 
        "--experiment={}".format(run_name), 
        "--input_shape={}".format(rc.get('detector_resolution')), 
        "--train_data_dir=fjlfbwjefrlbwelrfb_man_we_need_a_better_detector_codebase", 
        "--batch_size={}".format(rc.get('detection_training_batch_size')), 
        "--image_shape={}".format(dc.get('video_resolution')),
        "--epochs={}".format(epochs)]
        
        if import_datasets:
            import_datasets = quote(import_datasets)
            cmd.append("--import_datasets={}".format(import_datasets))
        
        job_id = jm.run(cmd, "train_detector")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
        
def post_detect_objects_job(dataset_name, run_name):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    if rc.exists:
        cmd = [python_path, "detect_csv.py",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--res={}".format(rc.get("detector_resolution")),
               "--conf={}".format(rc.get("confidence_threshold")),
               "--bs={}".format(rc.get("detection_batch_size"))]

        job_id = jm.run(cmd, "detect_objects")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
    
def post_visualize_detections_job(dataset_name, run_name, confidence_threshold, coords):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = [python_path, "visualize_detections.py",
               "--cmd=findvids",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--res={}".format(rc.get("detector_resolution")),
               "--conf={}".format(confidence_threshold),
               "--fps={}".format(dc.get('video_fps')),
               "--coords={}".format(coords)]

        job_id = jm.run(cmd, "visualize_detections")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)

def post_visualize_tracks_world_coordinates_job(dataset_name, run_name, videos):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    videos = quote(videos)
    
    cmd = [python_path, "visualize_tracking.py",
           "--dataset={}".format(dataset_name),
           "--run={}".format(run_name),
           "--videos={}".format(videos)]
    
    job_id = jm.run(cmd, "visualize_detections")
    if job_id:
        return (job_id, 202)
    else:
        return (NoContent, 503)
        
def post_detections_to_world_coordinates_job(dataset_name, run_name, make_videos):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = [python_path, "detections_world.py",
               "--cmd=findvids",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--make_videos={}".format(make_videos),
               "--ssdres={}".format(rc.get("detector_resolution")),
               "--vidres={}".format(dc.get('video_resolution')),
               "--kltres={}".format(dc.get('point_track_resolution'))
                ]

        job_id = jm.run(cmd, "detections_to_world")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)

def post_optimize_tracking_world_coordinates_job(csv_ground_truth_file, dataset_name, run_name, date, detection_id, class_name_conversion, visualize, patience, per_iteration):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        this_run_path = runs_path / "{dn}_{rn}".format(dn=dataset_name, rn=run_name)
        csv_path = this_run_path / "world_trajectory_gt.csv"
        
        try:
            gt = csv_ground_truth_file.decode('utf-8')
        except:
            return ("Could not parse .csv file as UTF-8", 400)
        else:
            with csv_path.open('w') as f:
                f.write(gt)
            
            cmd = [python_path, "tracking_world_optimization.py",
                   "--dataset={}".format(dataset_name),
                   "--run={}".format(run_name),
                   "--date={}".format(date),
                   "--gt_csv={}".format(csv_path),
                   "--det_id={}".format(detection_id),
                   "--gt_class_name_conversion={}".format(class_name_conversion),
                   "--visualize={}".format(visualize),
                   "--patience={}".format(patience),
                   "--per_iteration={}".format(per_iteration)]
            
            job_id = jm.run(cmd, "optimize_tracking_world_coordinates")
            if job_id:
                return (job_id, 202)
            else:
                return (NoContent, 404)
    else:
        s = dataset_name + '_' + run_name
        return (s, 404)

def post_tracking_world_coordinates_job(dataset_name, run_name, confidence_threshold, make_videos):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = [python_path, "tracking_world.py",
               "--cmd=findvids",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--conf={}".format(confidence_threshold),
               "--make_videos={}".format(make_videos)]

        job_id = jm.run(cmd, "tracking_world_coordinates")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
        
def post_tracking_pixel_coordinates_job(dataset_name, run_name, confidence_threshold, make_videos):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = [python_path, "tracking.py",
               "--cmd=findvids",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--ssdres={}".format(rc.get("detector_resolution")),
               "--vidres={}".format(dc.get('video_resolution')),
               "--kltres={}".format(dc.get('point_track_resolution')),
               "--conf={}".format(confidence_threshold),
               "--make_videos={}".format(make_videos)]

        job_id = jm.run(cmd, "tracking_pixel_coordinates")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
        
def post_all_tracks_as_zip_job(dataset_name, run_name, tracks_format, coords):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    
    cmd = [python_path, "tracks_formats.py",
           "--dataset={}".format(dataset_name),
           "--run={}".format(run_name),
           "--tf={}".format(tracks_format),
           "--coords={}".format(coords)]
   
    job_id = jm.run(cmd, "all_tracks_as_zip")
    if job_id:
        return (job_id, 202)
    else:
        return (NoContent, 503)
        
def post_summary_video_job(dataset_name, run_name, num_clips, clip_length):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    
    if (type(num_clips) == int) and (type(clip_length) == int):
    
        cmd = [python_path, "visualize_summary.py",
               "--dataset={}".format(dataset_name),
               "--run={}".format(run_name),
               "--n_clips={}".format(num_clips),
               "--clip_length={}".format(clip_length)]  
        
        job_id = jm.run(cmd, "summary_video")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 500)

def get_visualization_list(dataset_name, run_name, visualization_type):   
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    this_run_path = runs_path / "{dn}_{rn}".format(dn=dataset_name, rn=run_name)
    
    if visualization_type == "summary":
        videos = [this_run_path / "summary.mp4"]
    elif visualization_type == "detections_pixels":
        videos = (this_run_path / "detections").glob('*.mp4')
    elif visualization_type == "detections_world":
        videos = (this_run_path / "detections_world").glob('*.mp4')
    elif visualization_type == "tracks_pixels":
        videos = (this_run_path / "tracks").glob('*_tracks.mp4')
    elif visualization_type == "point_tracks":
        videos = (datasets_path / dataset_name / "klt").glob('*_klt.mp4')
    elif visualization_type == "world_tracking_optimization":
        videos = [this_run_path / "world_tracking_optimization.mp4"]
    elif visualization_type == "tracks_world":
        videos = (this_run_path / "tracks_world").glob("*_tracks.mp4")
    else:
        return (NoContent, 500)
    
    videos = list(videos)
    videos = [x for x in videos if x.is_file()]
    videos.sort()
    videos = [x.stem for x in videos]
    
    to_remove = ['_tracks', '_klt']
    for i,v in enumerate(videos):
        for tr in to_remove:
            if v.endswith(tr):
                videos[i] = v[:-len(tr)]
    
    return (videos, 200)

def get_visualization(dataset_name, run_name, visualization_type, video_name):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    video_name = quote(video_name)
    this_run_path = runs_path / "{dn}_{rn}".format(dn=dataset_name, rn=run_name)
    if visualization_type == "summary":
        video_path = this_run_path / "summary.mp4"
    elif visualization_type == "detections_pixels":
        video_path = this_run_path / "detections" / (video_name + '.mp4')
    elif visualization_type == "detections_world":
        video_path = this_run_path / "detections_world" / (video_name + '.mp4')
    elif visualization_type == "tracks_pixels":
        video_path = this_run_path / "tracks" / (video_name + '_tracks.mp4')
    elif visualization_type == "point_tracks":
        video_path = datasets_path / dataset_name / "klt" / (video_name + '_klt.mp4')
    elif visualization_type == "world_tracking_optimization":
        video_path = this_run_path / "world_tracking_optimization.mp4"
    elif visualization_type == "tracks_world":
        video_path = this_run_path / "tracks_world" / "{vn}_tracks.mp4".format(vn=video_name)
    else:
        return (NoContent, 500)
        
    if video_path.is_file():
        video_file = send_file(str(video_path), mimetype='video/mp4')
        return (video_file, 200)
    else:
        return (NoContent, 404)

def get_tracks(dataset_name, run_name, video_name, tracks_format, coords):
    dataset_name, run_name, video_name = map(quote, (dataset_name, run_name, video_name))
    
    val = None
    try:
        val = format_tracks(dataset_name, run_name, video_name, tracks_format, coords=coords)
    except FileNotFoundError:
        return (NoContent, 404)
    except ValueError:
        return (NoContent, 500)
    
    if val is None:
        return (NoContent, 500)
    else:
        return (val, 200)
        
def get_all_tracks(dataset_name, run_name, tracks_format, coords):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    zip_path = runs_path / "{}_{}".format(dataset_name, run_name) / "track_zips" / (tracks_format+'.zip')
    
    if coords == 'world':
        zip_path = zip_path.with_name(zip_path.stem + '_world.zip')
    
    if zip_path.is_file():
        return (send_file(str(zip_path), mimetype='application/zip'), 200)
    else:
        return (NoContent, 500)

def get_track_zip_list(dataset_name, run_name):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    
    found = []
    
    for coords in ['pixels','world']:
        for tracks_format in all_track_formats:
            zip_path = runs_path / "{}_{}".format(dataset_name, run_name) / "track_zips" / (tracks_format+'.zip')
            if coords == 'world':
                zip_path = zip_path.with_name(zip_path.stem + '_world.zip')
            
            if zip_path.is_file():
                found.append({'coords':coords, 'tracks_format':tracks_format})
    
    return (found, 200)

def get_list_of_runs(dataset_name):
    dataset_name = quote(dataset_name)
    
    runs = list(runs_path.glob(dataset_name + '_*'))
    runs.sort()
    
    # Run names and dataset names shouldn't contain underscore characters, but just in case
    runs = [left_remove(x.name, dataset_name + '_') for x in runs]
    
    if runs:
        return (runs, 200)
    else:
        return (NoContent, 404)
        
def get_list_of_videos(dataset_name):
    dataset_name = quote(dataset_name)
    
    vids = [x.stem for x in (datasets_path / dataset_name / "videos").glob('*.mkv')]
    vids.sort()
    
    if vids:
        return (vids, 200)
    else:
        return (NoContent, 404)
        
def post_world_tracking_config(dataset_name, run_name, world_tracking_config):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    path = runs_path / "{}_{}".format(dataset_name, run_name) / "world_tracking_optimization.pklz"
    
    try:   
        wtc = WorldTrackingConfig(world_tracking_config)
    except ValueError:
        return (NoContent, 400)
    else:
        save(wtc, str(path))
        return (NoContent, 200)
    
def get_world_tracking_config(dataset_name, run_name):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    path = runs_path / "{}_{}".format(dataset_name, run_name) / "world_tracking_optimization.pklz"

    if path.is_file():
        wtc = load(path)
        return (wtc.get_dict(), 200)
    else:
        return (NoContent, 404)
        
def get_world_calibration(dataset_name):
    dataset_name = quote(dataset_name)
    path = datasets_path / dataset_name / "calib.tacal"

    if path.is_file():
        content = path.read_text()
        
        return (content, 200)
    else:
        return (NoContent, 404)


def post_world_calibration(dataset_name, calib_text):
    dataset_name = quote(dataset_name)
    path = datasets_path / dataset_name / "calib.tacal"
    
    try:
        calib_text = calib_text.decode('utf-8')
    except:
        return (NoContent, 400)
    else:
    
        if validate_calibration(calib_text):
            
            with path.open('w') as f:
                f.write(calib_text)
            
            return (NoContent, 200)
        else:
            return (NoContent, 400)

def post_world_map(dataset_name, map_image, parameter_file):
    if map_image.content_type != 'image/png':
        return ("Map image has to be in png format.", 400)
    path = datasets_path / dataset_name / "map.png"
    map_image.save(str(path))
    path = datasets_path / dataset_name / "map.tamap"
    parameter_file.save(str(path))

def get_usb():
    usb = Path('/usb/')
    if usb.is_dir():
        gen = usb.glob('**')
        files = []
        
        for filepath in gen:
            files.append(filepath)
            
            if len(files) > 1000:
                files.append("... (too many to show)")
                break
        
        return (files, 200)
    else:
        return (NoContent, 404)

def make_app():
    mydir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(mydir)
    app = connexion.App(__name__, specification_dir=mydir)
    app.add_api(mydir + '/strudl.yaml')
    return app

@click.command()
@click.option("--port", default=80, help="Port number. Note that if this is changed and run from within docker, the docker run command needs to be changed to forward the correct port.")
def main(port):

    # Allows the host computer to remain responsive even while long-running and heavy processes are started by server
    os.nice(10) # nice :)
    
    # Start server based on YAML specification
    app = make_app()
    app.run(port=port)

if __name__ == '__main__':
    main()
    
