import gzip
import pickle

def save(data, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def load(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data
    
if __name__ == '__main__':
    from klt2 import Track
#    tracks = load('20170516_163607_4C86.pklz')
#    
#    for i, tr in enumerate(tracks):
#        for t, x, y, _ in tr:
#            tr2 = [int(t), int(x), int(y), int(_)]
#            tracks[i] = tr2
#    
#    save(tracks, '20170516_163607_4C86_2.pklz')

    tracks = load('20170516_163607_4C86_2.pklz')
    for i, tr in enumerate(tracks):
        print("{} {}".format(i, tr))
    
            
