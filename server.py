import connexion
from connexion import NoContent
from shlex import quote
from flask import send_from_directory, send_file
from glob import glob
from os.path import isdir, isfile
import os
import cv2
from random import choice
import subprocess
import click

from jobman import JobManager
from config import DatasetConfig, RunConfig
from folder import mkdir, datasets_path, runs_path
from classnames import set_classnames, get_classnames
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

jm = JobManager()

def get_progress(dataset_name, run_name):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    
    ds_path = "{dsp}{ds}/".format(dsp=datasets_path, ds=dataset_name)
    
    if isdir(ds_path):
        progress = dict()
        
        progress['has_config'] = isfile(ds_path + 'config.txt')
        if progress['has_config']: 
            dc = DatasetConfig(dataset_name)
            
        progress['has_mask'] = isfile(ds_path + 'mask.png')
        progress['has_classnames'] = isfile(ds_path + 'classes.txt')
        progress['has_calibration'] = isfile(ds_path + 'calib.tacal')
        
        progress['number_of_timestamp_logs'] = len(glob(ds_path + 'logs/*.log'))
        progress['number_of_videos'] = len(glob(ds_path + 'videos/*.mkv'))
        progress['training_frames_to_annotate'] = len(glob(ds_path + 'objects/train/*/*.jpg'))
        progress['training_frames_annotated'] = len(glob(ds_path + 'objects/train/*/*.txt'))
        
        progress['videos_with_point_tracks_computed'] = len(glob(ds_path + 'klt/*.pklz'))
        progress['videos_with_point_tracks_visualized'] = len(glob(ds_path + 'klt/*.mp4'))
        
        progress['all_runs'] = [x.split('/')[-1].split('_')[-1] for x in glob("{rp}{ds}_*".format(rp=runs_path, ds=dataset_name))]
        
        run_path = "{rp}{ds}_{rn}/".format(rp=runs_path, ds=dataset_name, rn=run_name)
        if isdir(run_path):
            progress['has_this_run'] = True
            
            rprogress = dict()
            rprogress['has_pretrained_weights'] = isfile('/data/ssd/weights_SSD300.hdf5')
            rprogress['videos_with_detected_objects'] = len(glob(run_path + 'csv/*.csv'))
            rprogress['videos_with_detected_objects_visualized'] = len(glob(run_path + 'detections/*.mp4'))
            rprogress['videos_with_detected_objects_in_world_coordinates'] = len(glob(run_path + 'detections_world/*.csv'))
            rprogress['videos_with_detected_objects_in_world_coordinates_visualized'] = len(glob(run_path + 'detections_world/*.mp4'))
            rprogress['stored_weight_files'] = len(glob(run_path + 'checkpoints/*.hdf5'))
            rprogress['videos_with_pixel_coordinate_tracks'] = len(glob(run_path + 'tracks/*.pklz'))
            rprogress['videos_with_pixel_coordinate_tracks_visualized'] = len(glob(run_path + 'tracks/*.mp4'))
            rprogress['videos_with_world_coordinate_tracks'] = len(glob(run_path + 'tracks_world/*.pklz'))
            rprogress['videos_with_world_coordinate_tracks_visualized'] = len(glob(run_path + 'tracks_world/*.mp4'))
            rprogress['has_optimized_world_tracking'] = isfile(run_path + 'world_tracking_optimization.pklz')
            rprogress['has_visualized_optimized_world_tracking'] = isfile(run_path + 'world_tracking_optimization.mp4')
            rprogress['has_world_tracking_ground_truth'] = isfile(run_path + 'world_trajectory_gt.csv')
            rprogress['track_zips'] = [x.split('/')[-1] for x in glob(run_path + 'track_zips/*.zip')]
            
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
            return send_file(impath, mimetype=mime)
    else:
        return impath

def get_annotation_slideshow(dataset_name):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        imsize = dc.get('video_resolution')
        outpath = "{dsp}{dn}/slideshow.mp4".format(dsp=datasets_path, dn=dataset_name)
        res = slideshow(dataset_name, imsize, outpath)
        
        if not res:
            return (NoContent, 404)
        else:
            vid = send_file(outpath, mimetype='video/mp4')
            return (vid, 200)
    else:
        return (NoContent, 404)
    

def post_annotation_annotation(dataset_name, image_number, video_name, annotation_set, annotation_text):
    dataset_name, video_name, annotation_set = map(quote, (dataset_name, video_name, annotation_set))
    
    annotation_text = annotation_text.decode('utf-8')
    if validate_annotation(annotation_text, dataset_name):   
        folder_path = "{dsp}{dn}/objects/{ans}/{vn}/".format(dsp=datasets_path, dn=dataset_name, vn=video_name, ans=annotation_set)
        if isdir(folder_path):
            file_path = "{fp}{imnum}.txt".format(fp=folder_path, imnum=image_number)
        
            with open(file_path, 'w') as f:
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
            
def post_dataset(dataset_name, class_names):
    dataset_name = quote(dataset_name)
    path = "{}{}/".format(datasets_path, dataset_name)
    mkdir(path)
    mkdir(path + 'videos')
    class_names = [quote(x.lower()) for x in class_names.split(',')]
    
    set_classnames(dataset_name, class_names)
    return (NoContent, 200)
    
def get_datasets():
    datasets = glob("{}*".format(datasets_path))
    datasets = [x.split('/')[-1] for x in datasets if isdir(x)]
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
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    if rc.set_data(run_config):
        rc.save()
        return (NoContent, 200)
    else:
        return ("Could not interpret run configuration. Is some required parameter missing?", 500)

def post_pretrained_weights(weights_file):
    path = '/data/ssd/weights_SSD300.hdf5'
    weights_file.save(path)
    if validate_pretrained_md5(path):
        return (NoContent, 200)
    else:
        os.remove(path)
        return ("File rejected", 400)
    
def post_mask(dataset_name, mask_image_file):
    dataset_name = quote(dataset_name)
    mask_tmp_path = "{}{}/mask_tmp.png".format(datasets_path, dataset_name)
    mask_path = "{}{}/mask.png".format(datasets_path, dataset_name)
    mask_image_file.save(mask_tmp_path)
    
    success = False
    try:
        # This is not really safe, but at least should protect from some completely broken image files
        im = cv2.imread(mask_tmp_path, -1)
        assert(im.shape[2] == 4)
        cv2.imwrite(mask_path, im)
        success = True
    except:
        success = False
        
    os.remove(mask_tmp_path)
    
    if success:
        return (NoContent, 200)
    else:
        try:
            os.remove(mask_path)
        except:
            pass
            
        return (NoContent, 500)
    
def get_mask(dataset_name):
    dataset_name = quote(dataset_name)
    mask_path = "{}{}/mask.png".format(datasets_path, dataset_name)
    if isfile(mask_path):
        mask_file = send_file(mask_path, mimetype='image/png')
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
    dataset_name = quote(dataset_name)
    path = quote(path)
    
    if logs_path is None:
        logs_path = path
        # Since 'path' probably contains a query, like ending with '*.mkv', this should be removed
        if not (logs_path[-1] == '/'):
            logs_path = logs_path.strip(logs_path.split('/')[-1])
    else:
        logs_path = quote(logs_path)
    
    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        resolution = dc.get('video_resolution')
        fps = dc.get('video_fps')
        cmd = ["python", "import_videos.py",
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
        cmd = ["python", "klt.py",
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
        
def post_prepare_annotations_job(dataset_name):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    
    if dc.exists:
        cmd = ["python", "annotation_preparation.py",
               "--dataset={}".format(dataset_name),
               "--num_ims={}".format(dc.get('images_to_annotate')),
               "--ims_per_vid={}".format(dc.get('images_to_annotate_per_video')),
               "--train_amount={}".format(dc.get('annotation_train_split'))]
        
        job_id = jm.run(cmd, "prepare_annotations")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
        
def post_autoannotate_job(dataset_name, epochs=75, resolution="(640,480,3)"):
    dataset_name = quote(dataset_name)
    dc = DatasetConfig(dataset_name)
    if dc.exists:
        cmd = ["python", "autoannotate.py",
               "--dataset={}".format(dataset_name),
               "--input_shape={}".format(resolution),
               "--image_shape={}".format(dc.get('video_resolution')),
               "--epochs={}".format(epochs)]
        
        job_id = jm.run(cmd, "autoannotate")
        if job_id:
            return (job_id, 202)
        else:
            return (NoContent, 503)
    else:
        return (NoContent, 404)
    
def post_train_detector_job(dataset_name, run_name):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = ["python", "training_script.py", 
        "--name={}".format(dataset_name), 
        "--experiment={}".format(run_name), 
        "--input_shape={}".format(rc.get('detector_resolution')), 
        "--train_data_dir=fjlfbwjefrlbwelrfb", 
        "--batch_size={}".format(rc.get('detection_training_batch_size')), 
        "--image_shape={}".format(dc.get('video_resolution'))]
        
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
        cmd = ["python", "detect_csv.py",
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
        cmd = ["python", "visualize_detections.py",
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
        
def post_detections_to_world_coordinates_job(dataset_name, run_name, make_videos):
    dataset_name = quote(dataset_name)
    run_name = quote(run_name)
    rc = RunConfig(dataset_name, run_name)
    dc = DatasetConfig(dataset_name)
    if rc.exists and dc.exists:
        cmd = ["python", "detections_world.py",
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
        this_run_path = "{rp}{dn}_{rn}/".format(rp=runs_path, dn=dataset_name, rn=run_name)
        csv_path = "{trp}world_trajectory_gt.csv".format(trp=this_run_path)
        
        try:
            gt = csv_ground_truth_file.decode('utf-8')
        except:
            return ("Could not parse .csv file as UTF-8", 400)
        else:
            with open(csv_path, 'w') as f:
                f.write(gt)
            
            cmd = ["python", "tracking_world_optimization.py",
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
        cmd = ["python", "tracking_world.py",
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
        cmd = ["python", "tracking.py",
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
    
    cmd = ["python", "tracks_formats.py",
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
    
        cmd = ["python", "visualize_summary.py",
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
    this_run_path = "{rp}{dn}_{rn}/".format(rp=runs_path, dn=dataset_name, rn=run_name)
    
    if visualization_type == "summary":
        video_path = "{trp}summary.mp4".format(trp=this_run_path)
    elif visualization_type == "detections_pixels":
        video_path = "{trp}detections/*.mp4".format(trp=this_run_path)
    elif visualization_type == "detections_world":
        video_path = "{trp}detections_world/*.mp4".format(trp=this_run_path)
    elif visualization_type == "tracks_pixels":
        video_path = "{trp}tracks/*_tracks.mp4".format(trp=this_run_path)
    elif visualization_type == "point_tracks":
        video_path = "{dsp}{dn}/klt/*_klt.mp4".format(dsp=datasets_path, dn=dataset_name)
    elif visualization_type == "world_tracking_optimization":
        video_path = "{trp}world_tracking_optimization.mp4".format(trp=this_run_path)
    elif visualization_type == "tracks_world":
        video_path = "{trp}tracks_world/*_tracks.mp4".format(trp=this_run_path)
    else:
        return (NoContent, 500)
    
    videos = glob(video_path)
    videos.sort()
    videos = [x.split('/')[-1][:-4] for x in videos]
    
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
    this_run_path = "{rp}{dn}_{rn}/".format(rp=runs_path, dn=dataset_name, rn=run_name)
    if visualization_type == "summary":
        video_path = "{trp}summary.mp4".format(trp=this_run_path)
    elif visualization_type == "detections_pixels":
        video_path = "{trp}detections/{vn}.mp4".format(trp=this_run_path, vn=video_name)
    elif visualization_type == "detections_world":
        video_path = "{trp}detections_world/{vn}.mp4".format(trp=this_run_path, vn=video_name)
    elif visualization_type == "tracks_pixels":
        video_path = "{trp}tracks/{vn}_tracks.mp4".format(trp=this_run_path, vn=video_name)
    elif visualization_type == "point_tracks":
        video_path = "{dsp}{dn}/klt/{vn}_klt.mp4".format(dsp=datasets_path, dn=dataset_name, vn=video_name)
    elif visualization_type == "world_tracking_optimization":
        video_path = "{trp}world_tracking_optimization.mp4".format(trp=this_run_path)
    elif visualization_type == "tracks_world":
        video_path = "{trp}tracks_world/{vn}_tracks.mp4".format(trp=this_run_path, vn=video_name)
    else:
        return (NoContent, 500)
        
    if isfile(video_path):
        video_file = send_file(video_path, mimetype='video/mp4')
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
    zip_path = "{rp}{dn}_{rn}/track_zips/{tf}.zip".format(rp=runs_path, dn=dataset_name, rn=run_name, tf=tracks_format)    
    
    if coords == 'world':
        zip_path = zip_path.replace('.zip', '_world.zip')
    
    if isfile(zip_path):
        return (send_file(zip_path, mimetype='application/zip'), 200)
    else:
        return (NoContent, 500)

def get_track_zip_list(dataset_name, run_name):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    
    found = []
    
    for coords in ['pixels','world']:
        for tracks_format in all_track_formats:
            zip_path = "{rp}{dn}_{rn}/track_zips/{tf}.zip".format(rp=runs_path, dn=dataset_name, rn=run_name, tf=tracks_format)
            if coords == 'world':
                zip_path = zip_path.replace('.zip', '_world.zip')
            
            if isfile(zip_path):
                found.append({'coords':coords, 'tracks_format':tracks_format})
    
    return (found, 200)
    

def get_list_of_runs(dataset_name):
    dataset_name = quote(dataset_name)
    
    runs = glob("{rp}{dn}_*".format(rp=runs_path, dn=dataset_name))
    runs.sort()
    runs = [x.split('/')[-1].split('_')[-1] for x in runs]
    
    if runs:
        return (runs, 200)
    else:
        return (NoContent, 404)
        
def get_list_of_videos(dataset_name):
    dataset_name = quote(dataset_name)
    
    vids = glob("{dsp}{dn}/videos/*.mkv".format(dsp=datasets_path, dn=dataset_name))
    vids.sort()
    vids = [x.split('/')[-1].strip('.mkv') for x in vids]
    
    if vids:
        return (vids, 200)
    else:
        return (NoContent, 404)
        
def post_world_tracking_config(dataset_name, run_name, world_tracking_config):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    path = "{rp}{dn}_{r}/world_tracking_optimization.pklz".format(rp=runs_path, dn=dataset_name, r=run_name)
    
    try:   
        wtc = WorldTrackingConfig(world_tracking_config)
    except ValueError:
        return (NoContent, 400)
    else:
        save(wtc, path)
        return (NoContent, 200)
    
def get_world_tracking_config(dataset_name, run_name):
    dataset_name, run_name = map(quote, (dataset_name, run_name))
    path = "{rp}{dn}_{r}/world_tracking_optimization.pklz".format(rp=runs_path, dn=dataset_name, r=run_name)
    
    if isfile(path):
        wtc = load(path)
        return (wtc.get_dict(), 200)
    else:
        return (NoContent, 404)
        
def get_world_calibration(dataset_name):
    dataset_name = quote(dataset_name)
    path = "{dsp}{ds}/calib.tacal".format(dsp=datasets_path, ds=dataset_name)
    
    if isfile(path):
        with open(path, 'r') as f:
            content = f.read()
        
        return (content, 200)
    else:
        return (NoContent, 404)


def post_world_calibration(dataset_name, calib_text):
    dataset_name = quote(dataset_name)
    path = "{dsp}{ds}/calib.tacal".format(dsp=datasets_path, ds=dataset_name)
    
    try:
        calib_text = calib_text.decode('utf-8')
    except:
        return (NoContent, 400)
    else:
    
        if validate_calibration(calib_text):
            
            with open(path, 'w') as f:
                f.write(calib_text)
            
            return (NoContent, 200)
        else:
            return (NoContent, 400)

def get_usb():
    if isdir('/usb/'):
        files = glob('/usb/**',recursive=True)
        return (files, 200)
    else:
        return (NoContent, 404)

@click.command()
@click.option("--port", default=80, help="Port number. Note that if this is changed and run from within docker, the docker run command needs to be changed to forward the correct port.")
def main(port):
    # Allows the host computer to remain responsive even while long-running and heavy processes are started by server
    import os
    os.nice(10)
    
    # Start server based on YAML specification
    app = connexion.App(__name__, specification_dir='./')
    app.add_api('strudl.yaml')
    app.run(port=port)

if __name__ == '__main__':
    main()
    
