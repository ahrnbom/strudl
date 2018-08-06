""" A module for accessing timestamps in the log files for each video """

from random import choice
from datetime import datetime
from glob import glob
import numpy as np
from os.path import isfile

from folder import datasets_path

def line_to_datetime(line):
    splot = line.split(' ')
            
    frame_number,year,month,day = map(int, splot[0:4])
    
    t_splot = splot[4].replace('.',':').split(':')
    hour,minute,second,millisecond = map(int, t_splot)
    
    frame_time = datetime(year, month, day, hour, minute, second, millisecond*1000)
    
    return frame_time, frame_number

class Timestamps(object):
    def __init__(self, dataset, cache_limit=16):
        self.file_cache = {}
        self.cache_limit = cache_limit # number of files to be kept in memory at once
        
        self.logs_path = "{dsp}{d}/logs/".format(dsp=datasets_path, d=dataset)
    
    def get_frame_number_given_vidname(self, t, vidname):
        logpath = "{lp}{v}.log".format(lp=self.logs_path, v=vidname)
        
        if isfile(logpath):
            times = self.make_times(logpath)
            dts = [abs( (t-x).total_seconds() ) for x in times]
            return np.argmin(dts)
    
        return None
        
    def get_frame_number(self, t, video_name_format=None, remove_last=0):
        """ Gets the video name and frame number from a datetime object (t). If the format for video names is provided, this is quite
           fast as only one log file is searched through. If no format is given, all log files are searched through which is slow. 
           It might be possible to use bisect search to speed this up also.
           """
        
        all_logs = glob("{lp}*.log".format(lp=self.logs_path))
        all_logs.sort()
        
        video_names = [x.split('/')[-1].strip('.log') for x in all_logs]
        
        logs = []
        vids = []
        if video_name_format is None:
            logs.extend(all_logs)
            vids.extend(video_names)
        else:
            start_times = [datetime.strptime(x[:-remove_last], video_name_format) for x in video_names]
            dts = [(t-x).total_seconds() for x in start_times]
            dts = [x if x > 0 else float('inf') for x in dts]
            index_min = np.argmin(dts)
            logs.append(all_logs[index_min])
            vids.append(video_names[index_min])
            
        best_dt = float("inf")
        best_frame_number = None
        best_video = None
        
        for i,l in enumerate(logs):
            times = self.make_times(l)
            dts = [abs( (t-x).total_seconds() ) for x in times]
            index_min = np.argmin(dts)
            closest_dt = dts[index_min]
            
            if closest_dt < best_dt:
                best_dt = closest_dt
                best_frame_number = index_min
                best_video = vids[i]
        
        return best_video, best_frame_number
        
    def get(self, video_name, frame_number=0):
        """ Gets the timestamp corresponding to a frame number in a video """
        log_path = "{lp}{vn}.log".format(lp=self.logs_path, vn=video_name)
        
        times = self.make_times(log_path)
        
        return times[frame_number]
   
    def make_times(self, log_path):
        if log_path in self.file_cache:
            times = self.file_cache[log_path]
        else:
            # Check if file cache is full, and if so, remove a random one
            while len(self.file_cache) > self.cache_limit:
                key = choice(tuple(self.file_cache.keys()))
                self.file_cache.pop(key)
            
            times = self._make_times(log_path)
            self.file_cache[log_path] = times
        
        return times

    def _make_times(self, log_path):
        with open(log_path, 'r') as f:
            lines = [x.strip('\n') for x in f.readlines()]
        
        times = []
        for line in lines:
            frame_time, frame_number = line_to_datetime(line)
            
            times.append(frame_time)
            
            assert(len(times)-1 == frame_number)
        
        return times

if __name__ == '__main__':
    ts = Timestamps('sweden2')
    print(ts.get('20170516_141545_4986', 20))
    
    print(ts.get_frame_number(datetime(2017, 5, 16, 6, 49, 38), '%Y%m%d_%H%M%S', 5))
    #print(ts.get_frame_number(datetime(2017, 5, 16, 6, 49, 38)))



