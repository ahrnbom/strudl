""" A module for loading tracks pklz files and converting them to some formats """

from os.path import isfile
from zipfile import ZipFile, ZIP_DEFLATED
from os import remove
from glob import glob
import click

from tracking import DetTrack
from tracking_world import WorldTrack
from storage import load
from folder import runs_path, mkdir
from util import print_flush

@click.command()
@click.option("--dataset", default="sweden2", help="Name of dataset")
@click.option("--run", default="default", help="Name of training run")
@click.option("--tf", type=click.Choice(['custom_text']))
@click.option("--coords", type=click.Choice(['pixels','world']), help="Pixel coordinates or world coordinates")
def generate_tracks_in_zip(dataset, run, tf, coords):
    tracks_format = tf
    if coords == 'pixels':
        tracks = glob("{rp}{dn}_{rn}/tracks/*.pklz".format(rp=runs_path, dn=dataset, rn = run))
    elif coords == 'world':
        tracks = glob("{rp}{dn}_{rn}/tracks_world/*.pklz".format(rp=runs_path, dn=dataset, rn = run))
    else:
        raise(ValueError("Incorrect coordinate system: {}".format(coords)))
        
    tracks.sort()
    
    tmp_path = "tmp.txt"
    
    zips_folder = "{rp}{dn}_{rn}/track_zips/".format(rp=runs_path, dn=dataset, rn=run)
    mkdir(zips_folder)
    
    zip_path = "{zf}{tf}.zip".format(zf=zips_folder, tf=tracks_format)
    if coords == 'world':
        zip_path = zip_path.replace('.zip', '_world.zip')
    
    with ZipFile(zip_path, mode='w', compression=ZIP_DEFLATED) as z:
        for t in tracks:
            tname = t.split('/')[-1]
            print_flush(tname)
            
            text = format_tracks_from_file(t, tracks_format, coords)
            with open(tmp_path, 'w') as f:
                f.write(text)
            
            z.write(tmp_path, arcname=tname.replace('.pklz', '.txt'))
    
    remove(tmp_path)
    print_flush("Done!")
    return zip_path

def format_tracks(dataset, run, video, tracks_format, coords='pixels'):
    t_foldername = "tracks"
    if coords == 'world':
        t_foldername = "tracks_world"
        
    tpath = "{rp}{d}_{r}/{tfn}/{v}_tracks.pklz".format(rp=runs_path, d=dataset, r=run, tfn=t_foldername, v=video)
    
    return format_tracks_from_file(tpath, tracks_format, coords)

def format_tracks_from_file(tpath, tracks_format, coords='pixels'):    
    if tracks_format == 'custom_text':    
        convert_track = convert_track_custom_text
    else:
        raise(ValueError('Tracks format {} is invalid'.format(tracks_format)))
        
    if isfile(tpath):
        tracks = load(tpath)
        text = ""
        
        tracks.sort(key=lambda x: x.id)
        
        for track in tracks:
            text += convert_track(track, coords) + '\n'
        
        return text
    else:
        raise(FileNotFoundError('Track {} not found'.format(tpath)))
        
    
        
def convert_track_custom_text(track, coords='pixels'):
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
    
    return text
        
if __name__ == '__main__':
    generate_tracks_in_zip()

