""" A module for accessing timestamps in the log files for each video """

from random import choice
from datetime import datetime
import numpy as np

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
        
        self.logs_path = datasets_path / dataset / "logs"
        
        self.start_times = None # The times when each log file starts, built by get_frame_number
    
    def get_frame_number_given_vidname(self, t, vidname):
        logpath = self.logs_path / (vidname+'.log')
        
        if logpath.is_file():
            times = self.make_times(logpath)
            dts = [abs( (t-x).total_seconds() ) for x in times]
            return np.argmin(dts)
    
        return None
        
    def get_frame_number(self, t):
        """ Gets the video name and frame number from a datetime object (t). 
        """
        
        all_logs = list(self.logs_path.glob('*.log'))
        all_logs.sort()
        
        video_names = [x.stem for x in all_logs]
        
        if self.start_times is None:
            self.start_times = dict()
            for video_name, log in zip(video_names, all_logs):
                with log.open('r') as f:
                    first_line = f.readline().rstrip()
                first_time, _ = line_to_datetime(first_line)
                self.start_times[video_name] = first_time
        
        # Find video which starts at or before t, as close as possible
        best_secs = float("inf")
        best_vid = None
        best_log = None
        for video_name, log in zip(video_names, all_logs):
            video_start = self.start_times[video_name]
            dt = t - video_start
            secs = dt.total_seconds()
            if secs >= 0:
                if secs < best_secs:
                    best_secs = secs
                    best_vid = video_name
                    best_log = log
        
        if not (best_vid is None):
            frame_num = self.get_frame_number_given_vidname(t, best_vid)
            return best_vid, frame_num
        else:
            return None, None
        
    def get_times(self, video_name):
        log_path = self.logs_path / "{vn}.log".format(vn=video_name)
        return self.make_times(log_path)

    def get(self, video_name, frame_number=0):
        """ Gets the timestamp corresponding to a frame number in a video """
        times = self.get_times(video_name)
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
        lines = log_path.read_text().split('\n')
        
        times = []
        for line in lines:
            if not line:
                continue
        
            frame_time, frame_number = line_to_datetime(line)
            
            times.append(frame_time)
            
            assert(len(times)-1 == frame_number)
        
        return times

if __name__ == '__main__':
    ts = Timestamps('sweden2')
    print(ts.get('20170516_141545_4986', 20))
    
    print(ts.get_frame_number(datetime(2017, 5, 16, 6, 49, 38)))



