""" A module for loading tracks .pklz files and converting them to some formats """

from zipfile import ZipFile, ZIP_DEFLATED
from os import remove
from glob import glob
import click

from tracking import DetTrack
from tracking_world import WorldTrack, WorldTrackingConfig
from storage import load
from folder import runs_path, mkdir
from util import print_flush

all_track_formats = ['csv','custom_text']

@click.command()
@click.option("--dataset", default="sweden2", help="Name of dataset")
@click.option("--run", default="default", help="Name of training run")
@click.option("--tf", type=str)
@click.option("--coords", type=click.Choice(['pixels','world']), help="Pixel coordinates or world coordinates")
def generate_tracks_in_zip(dataset, run, tf, coords):
    assert(tf in all_track_formats)    
    
    tracks_format = tf
    if coords == 'pixels':
        tracks = (runs_path / "{}_{}".format(dataset,run) / "tracks").glob('*.pklz')
    elif coords == 'world':
        tracks = (runs_path / "{}_{}".format(dataset,run) / "tracks_world").glob('*.pklz')
    else:
        raise(ValueError("Incorrect coordinate system: {}".format(coords)))
    
    tracks = list(tracks)        
    tracks.sort()
    
    zips_folder = runs_path / "{}_{}".format(dataset,run) / "track_zips"
    mkdir(zips_folder)
    
    zip_path = zips_folder / (tracks_format+'.zip')
    if coords == 'world':
        zip_path = zip_path.with_name(zip_path.stem + '_world.zip')
    
    with ZipFile(str(zip_path), mode='w', compression=ZIP_DEFLATED) as z:
        for t in tracks:
            tname = t.name
            print_flush(tname)
            
            text = format_tracks_from_file(t, tracks_format, coords)
            
            suffix = '.txt'
            if tracks_format == 'csv':
                suffix = '.csv'
            z.writestr(tname.replace('.pklz', suffix), text)
    
    print_flush("Done!")
    return zip_path

def format_tracks(dataset, run, video, tracks_format, coords='pixels'):
    t_foldername = "tracks"
    if coords == 'world':
        t_foldername = "tracks_world"
    
    tpath = runs_path / "{}_{}".format(dataset,run) / t_foldername / (video+'_tracks.pklz')    
    
    return format_tracks_from_file(tpath, tracks_format, coords)

def format_tracks_from_file(tpath, tracks_format, coords='pixels'):    
    if tracks_format == 'custom_text':    
        convert_track = convert_track_custom_text
    elif tracks_format == 'csv':
        convert_track = convert_track_csv
    else:
        raise(ValueError('Tracks format {} is invalid'.format(tracks_format)))
        
    if tpath.is_file():
        tracks = load(tpath)
        text = ""
        
        tracks.sort(key=lambda x: x.id)
        
        for i_track, track in enumerate(tracks):
            text += convert_track(track, coords, i_track)
        
        return text
    else:
        raise(FileNotFoundError('Track {} not found'.format(tpath)))      
    
def convert_track_csv(track, coords, i_track):
    text = ""
    if coords == 'pixels':
        # For csv files, this header should only be for the very first track
        if i_track == 0:
            text = "track_id,class,t,x,y,w,h\n"
        
        lines = {}
        
        for hist in track.history:
            t, x, y, w, h = hist[0:5]
            t, x, y, w, h = map(round, (t, x, y, w, h))
            t, x, y, w, h = map(int, (t, x, y, w, h))
            lines[t] = "{tid},{c},{t},{x},{y},{w},{h}\n".format(tid=track.id, c=track.c, t=t, x=x, y=y, w=w, h=h)
    elif coords == 'world':
        if i_track == 0:
            text = "track_id,class,t,frame_number,x,y,dx,dy,speed\n"
        
        lines = {}
        
        for hist in track.history:
            fn, t, x, y, dx, dy, speed = hist[0:7]
            fn, x, y, dx, dy, speed = map(lambda x: round(x,2), (fn, x, y, dx, dy, speed))
            lines[t] = "{tid},{c},{t},{fn},{x},{y},{dx},{dy},{sp}\n".format(tid=track.id, c=track.cn, t=t, fn=fn, x=x, y=y, dx=dx, dy=dy, sp=speed)
    else:
        raise(ValueError("Incorrect coords '{}'".format(coords)))
    
    keys = list(lines.keys())
    keys.sort()
    
    for key in keys:
        line = lines[key]
        text += line
    
    return text
        
def convert_track_custom_text(track, coords, i_track):
    if coords == 'pixels':
        text = "track ID: {tid}, class: {tc}\n".format(tid=track.id, tc=track.c)

        lines = {} # They are stored like this to keep only the last one for each frame
        
        for hist in track.history:
            t, x, y, w, h = hist[0:5]
            t, x, y, w, h = map(round, (t, x, y, w, h))
            t, x, y, w, h = map(int, (t, x, y, w, h))
            line = "  t:{t}, x:{x}, y:{y}, w:{w}, h:{h}\n".format(t=t, x=x, y=y, w=w, h=h)
            lines[t] = line
    elif coords == 'world':
        text = "track ID: {tid}, class: {tc}\n".format(tid=track.id, tc=track.cn)

        lines = {} # They are stored like this to keep only the last one for each frame
        
        for hist in track.history:
            fn, t, x, y, dx, dy, speed = hist[0:7]
            
            fn, x, y, dx, dy, speed = map(lambda x: round(x,2), (fn, x, y, dx, dy, speed))
            
            line = "  fn:{fn}, t:{t}, x:{x}, y:{y}, dx:{dx}, dy:{dy} sp:{speed}\n".format(t=t, x=x, y=y, dx=dx, dy=dy, fn=fn, speed=speed)
            lines[t] = line
    else:
        raise(ValueError("Incorrect coordinate system '{}'".format(coords)))
        
    keys = list(lines.keys())
    keys.sort()
    
    for key in keys:
        line = lines[key]
        text += line
    
    text += '\n'
    
    return text
        
if __name__ == '__main__':
    generate_tracks_in_zip()


