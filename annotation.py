""" A module for doing annotations. This module's functions work as a backend
    for the annotation Web UI.
"""

from pathlib import Path

from folder import mkdir, datasets_path, runs_path
from classnames import get_classnames
from visualize import class_colors
from util import to_hex, right_remove

def get_annotation_stats(dataset_name, annotation_set):
    # This is shown on the web UI to provide a sense of progress
    
    annot_folder = get_annotation_path(dataset_name, annotation_set)
    
    images = list(annot_folder.glob('*/*.jpg'))
    annots = list(annot_folder.glob('*/*.txt'))
    autos = list(annot_folder.glob('*/*.auto'))
    
    return (len(images), len(annots), len(autos))
    

def get_annotation_path(dataset_name, annotation_set, video_name=None, image_number=None, suffix='.txt'):
    # This function is called from server.py as well as from this module
    # It can give more or less detailed paths depending on options
    
    annot_folder = Path(datasets_path) / dataset_name / "objects" / annotation_set
    
    if video_name is None:
        return annot_folder
    else:
        annot_folder = annot_folder / video_name
        
        if image_number is None:
            return annot_folder
        else:
            file_path = annot_folder / "{imnum}{sfx}".format(imnum=image_number, sfx=suffix)
            
            if file_path.is_file():
                return file_path
            else:
                return None


def annotation_image_list(dataset_name, annotation_set):
    ds_path = get_annotation_path(dataset_name, annotation_set)
    
    vids = list(ds_path.glob('*'))
    if vids:
        vids = [x.name for x in vids if x.is_dir()]
        vids.sort()
        out = []
        
        for vid in vids:
            ims = list((ds_path / vid).glob('*.jpg'))
            ims.sort(key=lambda x: int(x.stem))
            for im in ims:
                imnum = im.stem
                txt = im.with_suffix('.txt')
                
                # Depending on the 'annotated' variable, the images show up in different colors in the web UI
                annotated = "not_annotated"
                if txt.is_file():
                    annotated = "already_annotated"
                else:
                    auto = im.with_suffix('.auto')
                    if auto.is_file():
                        annotated = "automatically_annotated"
                    
                out.append( (vid, imnum, annotated) )
        
        return out
    else:
        return None
        
def get_annotation_object(annots_path):
    """ Builds a JSON-serializable object (a list of dicts) for the annotations
        in an annotation file. Used when sending annotations to the Web UI.
    """

    lines = annots_path.read_text().split('\n')
    
    annots = []
    for line in lines:
        if not line:
            continue
    
        annot = {}
        splot = line.split(' ')
        annot['class_id'] = int(splot[0])
        annot['center_x'] = float(splot[1])
        annot['center_y'] = float(splot[2])
        annot['width'] = float(splot[3])
        annot['height'] = float(splot[4])
        annot['class_name'] = splot[-1]
        
        if splot[5].startswith('px:'):
            px = splot[5].strip('px:')
            py = splot[6].strip('py:')
            
            if not (px == 'auto'):
                px = px.split(',')
                py = py.split(',')
                annot['px'] = [float(x) for x in px]
                annot['py'] = [float(x) for x in py]
            else:
                annot['px'] = 'auto'
                annot['py'] = 'auto'
                
        elif splot[5].startswith('conf:'):
            annot['conf'] =  float(splot[5].split(':')[1])

        annots.append(annot)
    
    return annots
    
def annotation_data(dataset_name):
    """ Gets general annotation data for a dataset. Basically everything that the
        annotation Web UI needs, like classes and their colors, which keys to press
        for each class, and progress.
    """    
    
    try:
        classnames = get_classnames(dataset_name)
    except FileNotFoundError:
        return None
    else:
        # Removing R from this would not be sufficient, and would look like a bug
        all_keys = 'abcdefghijklmnopqrstuvwxyz'    
        
        colors = class_colors(len(classnames))
            
        out_colors = {}
        for cn, cc in zip(classnames, colors):
            out_colors[cn] = to_hex(cc) 
        
        keys_list = []
        keys = set()
        for cn in classnames:
            cn = cn.lower()
            success = False
            for letter in cn:
                if not (letter in keys):      
                    if not letter == 'r': # 'R' is not okay, since that button is used for cancelling
                        keys_list.append(letter)
                        keys.add(letter)
                        success = True
                        break
            
            while not success:
                letter = choice(all_keys)
                if not (letter in keys):
                    if not letter == 'r': # 'R' is not okay, since that button is used for cancelling
                        keys.add(letter)
                        keys_list.append(letter)
                        success = True
        
        key_codes_list = [ord(x)-32 for x in keys_list]  # convert to nonsense Javascript keyCodes          
        keys_list = [x.upper() for x in keys_list] # are nicer visualized in upper case in the Web UI
        
        train_stats = get_annotation_stats(dataset_name, 'train')
        test_stats = get_annotation_stats(dataset_name, 'test')
        
        out = [classnames, out_colors, keys_list, key_codes_list, train_stats, test_stats]
        
        return out
