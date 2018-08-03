""" A module for doing annotations. This module's functions work as a backend
    for the annotation Web UI.
"""

from glob import glob
from os.path import isfile, isdir

from folder import mkdir, datasets_path, runs_path
from classnames import get_classnames
from visualize import class_colors
from util import to_hex

def get_annotation_stats(dataset_name, annotation_set):
    annot_folder = get_annotation_path(dataset_name, annotation_set)
    
    images = glob(annot_folder + '*/*.jpg')
    annots = glob(annot_folder + '*/*.txt')
    autos = glob(annot_folder + '*/*.auto')
    
    return (len(images), len(annots), len(autos))
    

def get_annotation_path(dataset_name, annotation_set, video_name=None, image_number=None, suffix='.txt'):
    annot_folder = "{dsp}{dn}/objects/{ans}/".format(dsp=datasets_path, dn=dataset_name, ans=annotation_set)
    
    if video_name is None:
        return annot_folder
    else:
        annot_folder += "{vn}/".format(vn=video_name)
        
        if image_number is None:
            return annot_folder
        else:
            file_path = annot_folder + "{imnum}{sfx}".format(imnum=image_number, sfx=suffix)
            
            if isfile(file_path):
                return file_path
            else:
                return None


def annotation_image_list(dataset_name, annotation_set):
    ds_path = get_annotation_path(dataset_name, annotation_set)
    
    vids = glob(ds_path + '*')
    if vids:
        vids = [x.split('/')[-1] for x in vids if isdir(x)]
        vids.sort()
        out = []
        
        for vid in vids:
            ims_path = "{ds_path}{vid}/*.jpg".format(ds_path=ds_path, vid=vid)

            ims = glob(ims_path)
            ims.sort(key=lambda x: int(x.split('/')[-1].strip('.jpg')))
            for im in ims:
                imnum = im.split('/')[-1].strip('.jpg')
                txt = im.replace('.jpg', '.txt')
                annotated = "not_annotated"
                if isfile(txt):
                    annotated = "already_annotated"
                else:
                    auto = im.replace('.jpg','.auto')
                    if isfile(auto):
                        annotated = "automatically_annotated"
                    
                out.append( (vid, imnum, annotated) )
        
        return out
    else:
        return None
        
def get_annotation_object(impath):
    with open(impath, 'r') as f:
        lines = [x.strip('\n') for x in f.readlines()]
    
    annots = []
    for line in lines:
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
    try:
        classnames = get_classnames(dataset_name)
    except FileNotFoundError:
        return None
    else:
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
