""" A module for validating various kinds of data """

from classnames import get_classnames
from hashlib import md5

from folder import ssd_path


def validate_pretrained_md5(filepath):
    expected_hash = '9ae4b93e679ea30134ce37e3096f34fa'

    with open(filepath, 'rb') as f:
        data = f.read()
        if md5(data).hexdigest() == expected_hash:
            return True
    
    return False

def validate_calibration(text):
    legal_params = {'dx','dy','Cx','Cy','Sx','f','k','Tx','Ty','Tz','r1','r2','r3','r4','r5','r6','r7','r8','r9'}
    
    try:
        lines = text.split('\n')
        assert(len(lines) == len(legal_params))
        for line in lines:
            splot = line.split(':')
            splot = [x.strip() for x in splot]
            
            assert(len(splot) == 2)
            assert(splot[0] in legal_params)
            float(splot[1])
    except:
        return False
    else:
        return True

def validate_logfile(path):
    try:
        with open(path, 'r') as f:
            lines = [x.strip('\n') for x in f.readlines()]

        for i, line in enumerate(lines):
            splot = line.split(' ')
            
            assert(i == int(splot[0]))
            int(splot[1])
            int(splot[2])
            int(splot[3])
            
            timestamp = splot[4].replace('.',':').split(':')
            assert(len(timestamp) == 4)
            for ts in timestamp:
                int(ts)
    except:
        return False

    return True

def validate_annotation(text, dataset):
    classnames = get_classnames(dataset)    
    
    try:
        lines = text.split('\n')
        for line in lines:
            if line.isspace() or (not line):
                continue
                
            splot = line.split(' ')

            int(splot[0])
            float(splot[1])        
            float(splot[2])
            float(splot[3])
            float(splot[4])
            
            assert(splot[5][0:3] == "px:")
            assert(splot[6][0:3] == "py:")
            
            px = splot[5].strip('px:')
            py = splot[6].strip('py:')
            
            if not (px == 'auto'):
                px = px.split(',')       
                assert(len(px)==4)
                for x in px:
                    float(x)
                    
            if not (py == 'auto'):
                py = py.split(',')
                assert(len(py)==4)
                for y in py:
                    float(y)
            
            assert(splot[7] in classnames)
    except:
        return False
        
    return True
        
if __name__ == '__main__':
    # readme.md suggests running this, so don't any other tests here.
    if validate_pretrained_md5(ssd_path + '/weights_SSD300.hdf5'):
        print("Weights file found and is OK!")
    else:
        print("Weight file found, but is possibly broken.")


