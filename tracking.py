import pandas as pd
import numpy as np
from random import choice
from itertools import combinations, count
import click

from util import parse_resolution, print_flush
from apply_mask import Masker

class Config(object):
    """ 
    Configuration class for tracking. This exists mainly to have all configuration parameters
    in a single place.
    """

    def __init__(self, vidres, kltres, confthresh):
        """
        Arguments:
        vidres -- video resolution
        kltres -- resolution of coordinates of KLT point tracks
        
        """
        self.vidres = vidres
        self.kltres = kltres
        self.klt_x_factor = self.vidres[0]/self.kltres[0]
        self.klt_y_factor = self.vidres[1]/self.kltres[1]
        
        self.confidence_thresh = confthresh
        
        self.sq_distance_thresh = 1000
        self.dist_time_factor = 1.05
        
        self.aspectratio_thresh = 0.3
        self.ar_time_factor = 1.1
        
        self.klt_drop_assign_thresh = 3 # if this many klt tracks would have been dropped by new detection assignment, it is not assigned
        
        # Larger objects should more likely be mapped to "dissimilar" existing tracks
        self.size_dist_factor = 0.1
        self.size_ar_factor = 0.1
        
        self.score_dist_weight = 1.
        self.score_ar_weight = 1.
        self.score_klt_weight = 1.
        
        self.lost_thresh = 5 # frames
        
        self.corner_thresh = 3 # pixels
        
        self.iou_thresh = 0.5
        self.iou_minlength = 4 # tracks shorter than this do no "kill" other tracks by overlapping
        
        # Number of frames tracks can survive without detections. This varies for classes, depending on how good we think the detector is
        self.nodet_lifetimes = {'car': 20, 'bus': 5, 'person':50, 'bicycle':50, 'motorbike':5}
        
        self.min_track_length = 4
        
id_maker = count()

class DetTrack(object):
    """
    A single tracked object. 
    """
    def __init__(self, x, y, w, h, c, t):
        """
        Creates a track, assumes it is created by a detected object.
        
        Arguments:
        x -- initial x position
        y -- initial y position
        w -- initial width
        h -- initial height
        c -- object class
        t -- initial time 
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c # object class, like "car"
        self.t = t # the time when position etc. was last updated
        self.last_detection = t # the time when track was updated by detection
        self.dets = []
        self.klts = []
        self.klt_indices = set()
        self.history = []
        self.klt_checkpoints = {}
        self.drop_reason = "not dropped"
        self.id = next(id_maker)
        
        self.store()
        
    def update(self, x, y, w, h, t, by_detection=True):   
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t = t  
        if by_detection:
            self.last_detection = t
        
        self.store(by_update=True)
        
    def move(self, dx, dy, t):
        self.x += dx
        self.y += dy
        self.t = t 
            
        self.store()
        
    def store(self, by_update=False):
        self.history.append( (self.t, self.x, self.y, self.w, self.h, by_update) )
        
    def length(self):
        if len(self.history) < 2:
            return 0
            
        mint = self.history[0][0]
        maxt = self.history[-1][0]
                
        return maxt-mint
        
    def clean(self):
        # Reduce memory consumption when stored
        self.dets = None
        self.klts = None
        

def inside(box, x, y):
    return (x < (box.x + box.w)) and (x > box.x) and (y < (box.y + box.h)) and (y > box.y)

def overlap(a, b):
    """ Quick check if two AABBs overlap """
    axs = [a.x, a.x+a.w]
    ays = [a.y, a.y+a.h]
    bxs = [b.x, b.x+b.w]
    bys = [b.y, b.y+b.h]
    
    for ax in axs:
        for ay in ays:
            if inside(b, ax, ay):
                return True
                
    for bx in bxs:
        for by in bys:
            if inside(a, bx, by):
                return True
    
    return False

def iou(a, b): 
    """ Computes intersection over union, which is high if two AABBs are similar and highly overlapping """ 
    a_xmax = a.x + a.w
    a_ymax = a.y + a.h
    b_xmax = b.x + b.w
    b_ymax = b.y + b.h
    
    xA = max(a.x, b.x)
    yA = max(a.y, b.y)
    xB = min(a_xmax, b_xmax)
    yB = min(a_ymax, b_ymax)
	
    interArea = (xB - xA) * (yB - yA)
	
    boxAArea = a.w*a.h
    boxBArea = b.w*b.h
	
    iou = interArea / float(boxAArea + boxBArea - interArea)
	
    return iou

def convert_klt(old_klt, config):
    """ 
    Converts KLT tracks to a more convenient format, 
    where each track is a dict, allowing fast access to 
    its position on a given time. 
    """
    new_klt = []
    
    ids_per_frame = {}
    
    for klt_id, old_track in enumerate(old_klt):
        new_track = {}
        new_track['id'] = klt_id
        new_track['taken'] = False
        for klt_point in old_track:
            frame_num = klt_point[0]
            x = config.klt_x_factor*klt_point[1]
            y = config.klt_y_factor*klt_point[2]
            
            new_track[frame_num] = (x, y)
            
            if not frame_num in ids_per_frame:
                ids_per_frame[frame_num] = []
            
            ids_per_frame[frame_num].append(klt_id)
        
        new_klt.append(new_track)
    
    return new_klt, ids_per_frame

def find_klt_for_frame(klt, klt_frames, i):
    """ Finds all KLT tracks appearing in a given frame """
    if not i in klt_frames:
        return []
    
    ids = klt_frames[i]
    klt_this = [klt[x] for x in ids]
            
    return klt_this

def build_tracks(detections, klt, klt_frames, config):
    """ 
    Main entry point for tracking. Builds tracks by going through each frame's detections and KLT tracks
    
    Arguments:
    detections -- data frame with detections for all frames in a video
    klt        -- list of KLT tracks, assumed to be converted by convert_klt
    klt_frames -- precomputed dict of which KLT track IDs appear every frame, from convert_klt
    config     -- a Config object
    """
    tracks = []
    lost_tracks = []
    
    n_frames = max(detections.frame_number)
    
    for i in range(n_frames):
        
        klt_this = find_klt_for_frame(klt, klt_frames, i)
        tracks, lost_tracks = update_tracks(i, tracks, lost_tracks, klt_this, config)        
        
        det_this = detections[detections.frame_number == i]
        tracks = assign_detections(i, tracks, det_this, config)
        
    lost_tracks.extend(tracks)
    tracks = remove_short_tracks(lost_tracks, config.min_track_length)
    
    for track in tracks:
        track.clean()
    
    return tracks
              
def update_tracks(i, tracks, lost_tracks, klt_this, config):
    # Drop tracks that are not up-to-date
    to_lose = []
    for i_track, track in enumerate(tracks):
        dt = i - track.t
        if dt >= config.lost_thresh:
            to_lose.append(i_track)
            track.drop_reason = "too old"
    
    # Drop tracks that have reached the edges of the image
    for i_track, track in enumerate(tracks):
        if (track.x - track.w/2 < config.corner_thresh) \
         or (track.y - track.h/2 < config.corner_thresh) \
         or (track.x + track.w/2 > config.vidres[0]-config.corner_thresh) \
         or (track.y + track.h/2 > config.vidres[1]-config.corner_thresh):
            to_lose.append(i_track)
            track.drop_reason = "reached edge"
    
    # Drop tracks that overlap significantly
    for t1,t2 in combinations(enumerate(tracks),2):
        i_track, track1 = t1
        j_track, track2 = t2
        
        if not overlap(track1, track2):
            continue
            
        tracks_iou = iou(track1, track2)
        if tracks_iou > config.iou_thresh:
            # If one of the tracks is too young, only kill that one
            t1l = track1.length()
            t2l = track2.length()
            
            if (t1l <= config.iou_minlength) and (t2l > config.iou_minlength):
                to_lose.append(i_track)
                track1.drop_reason = "overlap with {}, too young".format(track2.id)
            elif (t2l <= config.iou_minlength) and (t1l > config.iou_minlength):
                to_lose.append(j_track)
                track2.drop_reason = "overlap with {}, too young".format(track1.id)
            else:        
                to_lose.append(i_track)
                to_lose.append(j_track)
                track1.drop_reason = "overlap with {}".format(track2.id)
                track2.drop_reason = "overlap with {}".format(track1.id)
                
    # Drop tracks that have gone too long without detections
    for i_track, track in enumerate(tracks):
        dt = i - track.last_detection
        thresh = config.nodet_lifetimes[track.c]
        if dt >= thresh:
            to_lose.append(i_track)
            track.drop_reason = "no detections"
    
    to_lose = list(set(to_lose)) # Remove duplicates, cannot remove same element twice
    to_lose.sort()    
    for track_index in reversed(to_lose):
        track = tracks.pop(track_index)
        lost_tracks.append(track)
    
    for track in tracks:
        klt_update(i, track, klt_this)
    
    return tracks, lost_tracks

def klt_update(i, track, klt_this):
    # Given a track, look for klt points inside the box, and find where they seem to be going, updating x, y
    
    # Remove klt candidate points that are already in track
    klt_candidates = [k for k in klt_this if (not k['taken']) and (not k['id'] in track.klt_indices)]
    
    xmin = track.x - track.w/2
    ymin = track.y - track.h/2
    xmax = track.x + track.w/2
    ymax = track.y + track.h/2
    
    # Remove KLT points that are no longer inside the box
    to_remove = []
    for ik_track, k_track in enumerate(track.klts):
        if i in k_track:
            k_point = k_track[i]
            
            kx = k_point[0]
            ky = k_point[1]
            
            if (kx < xmin) or (kx > xmax) or (ky < ymin) or (ky > ymax):
                to_remove.append( (ik_track, k_track) )
        else:
            # Maybe not? 
            to_remove.append( (ik_track, k_track) )
    
    for klts_index, klt_track in reversed(to_remove):
        track.klt_indices.remove(klt_track['id'])
        track.klts.pop(klts_index)
        klt_track['taken'] = False
            
    # Add fitting candidates            
    for k_track in klt_candidates:            
        k_point = k_track[i]
        kx = k_point[0]
        ky = k_point[1]
        if kx > xmin and kx < xmax and ky > ymin and ky < ymax:
            track.klt_indices.add(k_track['id'])
            track.klts.append(k_track)
            k_track['taken'] = True
    
    # Find movement of all KLT tracks at this time
    dxs = []
    dys = []
    
    klt_checkpoint = []
    for i_ktrack, k_track in enumerate(track.klts):
        if (i in k_track) and ((i-1) in k_track):
            this = k_track[i]
            prev = k_track[i-1]
                
            klt_checkpoint.append( (k_track['id'], this) )
            
            dx = this[0]-prev[0]
            dy = this[1]-prev[1]
            
            dxs.append(dx)
            dys.append(dy)
    
    if dxs:        
        dx = np.mean(dxs)
        dy = np.mean(dys)
        
        track.move(dx, dy, i)
            
    track.klt_checkpoints[i] = klt_checkpoint
        
def assign_detections(i, tracks, dets, config):
    new_tracks = []
    for i_det, d in dets.iterrows():
        if d.confidence > config.confidence_thresh:
        
            # Skip detections that are near the edges
            if (d.xmin < config.corner_thresh) \
             or (d.ymin < config.corner_thresh) \
             or (d.xmax > config.vidres[0]-config.corner_thresh) \
             or (d.ymax > config.vidres[1]-config.corner_thresh):
                continue
                
            # Check if any existing track matches this object
            x = (d.xmin + d.xmax)/2
            y = (d.ymin + d.ymax)/2
            
            w = (d.xmax - d.xmin)
            h = (d.ymax - d.ymin)
            ar = w/h
            area = w*h
            
            c = d.class_name
            
            candidate_tracks = []
            
            for track in tracks:
                if track.c == c:
                    dx = x - track.x
                    dy = y - track.y
                    sqdist = dx*dx + dy*dy
                    
                    dt = i - track.t
                    
                    dist_thresh = config.sq_distance_thresh + dt * config.dist_time_factor + area*config.size_dist_factor
                    dist_score = sqdist / dist_thresh
                    
                    if dist_score < 1.:
                        ar_thresh = config.aspectratio_thresh + dt*config.ar_time_factor + area*config.size_ar_factor
                        
                        track_ar = track.w / track.h
                        
                        ar_score = abs(track_ar - ar) / ar_thresh
                        if ar_score < 1.:
                            # Seems like a match
                            
                            # Before computing score, see if we would drop any KLT tracks with new detection and include that in scoring
                            klts = track.klts
                            nklt = 0
                            n_still_inside = 0
                            for klt in klts:
                                if i in klt:
                                    nklt += 1
                                    
                                    k = klt[i]
                                    if (k[0] > d.xmin) and (k[0] < d.xmax) and (k[1] > d.ymin) and (k[1] < d.ymax):
                                        n_still_inside += 1
                            
                            # Is this a smart way to compute this score?
                            klt_score = nklt - n_still_inside
                            
                            if klt_score < config.klt_drop_assign_thresh:
                                score = config.score_dist_weight*dist_score + config.score_ar_weight*ar_score + config.score_klt_weight*klt_score
                                candidate_tracks.append( (track, score) )
            
            ncand = len(candidate_tracks)
            if ncand == 0:
                track = DetTrack(x, y, w, h, c, i)
                new_tracks.append(track)
            else:
                best_score = float("inf")
                candidate = None
                for candidate, score in candidate_tracks:
                    if score < best_score:
                        best_score = score
                        best = candidate
                
                track = candidate
                track.update(x, y, w, h, i)
                track.dets.append(d)
                
    tracks.extend(new_tracks)
    return tracks     
    
def remove_short_tracks(tracks, minimum):
    return [x for x in tracks if x.length() >= minimum]

@click.command()
@click.option("--cmd", default="findvids", help="Which command to run, 'findvids' to look for videos to track on, or else a path to an input video")
@click.option("--dataset", default="sweden2", help="Which dataset to use")
@click.option("--run", default="default", help="Which training run to use")
@click.option("--vidres", default="(640,480,3)", help="Resolution of the videos, like '(640,480,3)' with width, height and then number of channels")
@click.option("--ssdres", default="(640,480,3)", help="Resolution images fed into object detector, like '(640,480,3)' with width, height and then number of channels")
@click.option("--kltres", default="(320,240)", help="Resolution of images used for point tracking, like '(320, 240)' with width and then height")
@click.option("--conf", default=0.8, type=float, help="Confidence threshold")
@click.option("--make_videos", default=True, type=bool, help="If true, videos are generated")
def main(cmd, dataset, run, vidres, ssdres, kltres, conf, make_videos):
    from storage import load, save
    from folder import datasets_path, runs_path
    
    mask = Masker(dataset)
    #v = '20170516_163607_4C86'
    #v = '20170516_121024_A586'
    
    if cmd == "findvids":
        from glob import glob
        vidnames = glob('{}{}/videos/*.mkv'.format(datasets_path, dataset))
        vidnames = [x.split('/')[-1].strip('.mkv') for x in vidnames]
        vidnames.sort()
        
        outfolder = '{}{}_{}/tracks/'.format(runs_path, dataset, run)
    else:
        vidnames = [cmd]
        outfolder = './'
    
    vidres = parse_resolution(vidres)
    ssdres = parse_resolution(ssdres)
    kltres = parse_resolution(kltres)
    
    x_factor = float(vidres[0])/ssdres[0]
    y_factor = float(vidres[1])/ssdres[1]
    det_dims = ('xmin', 'xmax', 'ymin', 'ymax')
    det_factors = (x_factor, x_factor, y_factor, y_factor)
    
    c = Config(vidres, kltres, conf)
    
    from folder import mkdir
    mkdir(outfolder)
    
    for v in vidnames:    
        detections = pd.read_csv('{}{}_{}/csv/{}.csv'.format(runs_path, dataset, run, v))
        for dim, factor in zip(det_dims, det_factors):
            detections[dim] = round(detections[dim]*factor).astype(int)
            
        klt = load('{}{}/klt/{}.pklz'.format(datasets_path, dataset, v))
        klt, klt_frames = convert_klt(klt, c)
        
        tracks = []
        if len(detections)>0:
            tracks = build_tracks(detections, klt, klt_frames, c)
            print_flush("{}  tracks done".format(v))
            save(tracks, '{}{}_tracks.pklz'.format(outfolder, v))
        else:
            print_flush("{}  skipping tracking, because there were no detections".format(v))
        
        if make_videos:
            if tracks:
                from visualize_tracking import render_video
                vidpath = "{}{}/videos/{}.mkv".format(datasets_path, dataset, v)
                render_video(tracks, vidpath, "{}{}_tracks.mp4".format(outfolder, v), mask=mask)
                print_flush("{}  video done".format(v))
            else:
                print_flush("{}  skipping video rendering, because there were no tracks".format(v))
    
    print_flush("Done!")

if __name__ == '__main__':
    main()

    
    

