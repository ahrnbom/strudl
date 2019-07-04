""" A simple job manager, to be user by the server for long-running jobs.
    For now, only one job can be running at the time.
    Each job has a unique ID, which is presented in the Web UI, containing
    the time when the job was first requested, the type of job and some randomness.
    A background thread is looking for new jobs to run every 5 seconds, so that
    a new job from the queue is automatically started once the current job finishes.
    This module also takes care of the log files created for each job.
"""

import subprocess
import threading
from uuid import uuid4
from time import time
from datetime import datetime

from folder import mkdir, jobs_path

def new_id(job_type=""):
    return "{ts}_{jt}_{rand}".format(ts=timestamp('-','-','_'), jt=job_type, rand=str(uuid4())[0:6])
    
def timestamp(sep1='-', sep2=':', sep3=' '):
    return datetime.fromtimestamp(time()).strftime('%Y{sep1}%m{sep1}%d{sep3}%H{sep2}%M{sep2}%S'.format(sep1=sep1, sep2=sep2, sep3=sep3))

class Worker(threading.Thread):
    jm = None
    daemon = True
    
    def set_jobmanager(self, jm):
        self.jm = jm
    
    def run(self):
        e = threading.Event()
        while True:
            # Wait 5 seconds (to prevent a tight loop)
            # Because of this, it's often necessary to press Ctrl + C twice to kill
            # server.py. As far as I know, there's no other problems related to this...
            # But it would be nice if there was some other way.
            e.wait(5) 
            
            if self.jm.is_available():
                job = None
                with self.jm.queue_lock:     
                    if self.jm.queue:
                        job = self.jm.queue.pop(0)
                
                if job is not None:
                    job_id, cmd, job_type = job
                    self.jm.actually_run(cmd, job_id, job_type)
                        
class JobManager(object):
    current = None
    history = []
    queue_lock = threading.RLock()
    queue = []
    worker = None
    
    def __init__(self):  
        mkdir(jobs_path)
        
    def start(self):
        # There's no point in starting the background thread until there's a job to run.
        if self.worker is None:
            self.worker = Worker()
            self.worker.set_jobmanager(self)
            self.worker.start()
    
    def is_available(self):
        if (self.current is None):
            return True
        else: 
            if self.current.poll() is None:
                # Still running
                return False
            else:
                return True
    
    def stop(self, job_id, method='terminate'):
        """ Can be used to cancel a currently running job or remove it from the queue.
            There are buttons in the Web UI that call this.
        """
        if self.history[-1][0] == job_id:
            if method == 'terminate':
                self.current.terminate()
            elif method == 'kill':
                self.current.kill()
            else:
                raise(ValueError("Incorrect method {}".format(method)))
            
            self.current = None
            return True
            
        else:
            did_remove = False
            with self.queue_lock:
                to_remove = None
                for q in self.queue:
                    if q[0] == job_id:
                        to_remove = q
                        
                if not (to_remove is None):
                    self.queue.remove(to_remove)
                    did_remove = True
                    
            return did_remove
    
    def run(self, cmd, job_type=""):
        job_id = new_id(job_type)
        
        with self.queue_lock:
            self.queue.append( (job_id, cmd, job_type) )
    
        self.start()
        
        return job_id
    
    def actually_run(self, cmd, job_id, job_type=""):
        self.history.append((job_id, ' '.join(cmd)))
        
        logfile = jobs_path / (job_id + '.log')

        with logfile.open('w') as out:
            out.write("Command: {}\n".format(cmd))
            out.write("{}\n".format(timestamp()))
            self.current = subprocess.Popen(cmd, stdout=out, stderr=out, bufsize=0)

    def get_jobs(self, job_type):
        """ There are several variations of this:
            recent: All jobs since the creation of this JobManager, including running ones.
            recent_with_status: Same as recent but with info about if they're running/done/queued
            all: All jobs it can find, including those before the creation of this JobManager
            running: Any job currently running
            queued: All jobs in the queue
        """
        if job_type == 'recent':
            return [x[0] for x in self.history]
        elif job_type == 'recent_with_status':
            job_ids = [x[0] for x in self.history]
            
            out = []
            for job_id in job_ids:
                if (not self.is_available()) and (self.history[-1][0] == job_id):
                    result = "running"
                else:
                    log_path = jobs_path / (job_id + '.log')
                    if log_path.is_file():
                        lines = log_path.read_text().split('\n')
                        
                        result = "failure"
                        for line in lines:
                            if line == "Done!":
                                result = "success"
                    else:
                        result = "failure"
                    
                out.append({"id":job_id, "result":result})
            
            with self.queue_lock:        
                queue_ids = [x[0] for x in self.queue]
                
            for queue_id in queue_ids:
                out.append({"id":queue_id, "result":'queued'})
                
            return out
                
        elif job_type == 'all':
            logs = list(jobs_path.glob('*.log'))
            job_ids = [x.stem for x in logs]
            job_ids.sort()
            
            with self.queue_lock:
                queue_ids = [x[0] for x in self.queue]
                
            job_ids.extend(queue_ids)
            
            return job_ids
        elif job_type == 'running':
            if not self.is_available():
                return self.history[-1][0]
            else:
                return []
        elif job_type == 'queued':
            with self.queue_lock:
                queued = [x[0] for x in self.queue]
            return queued
        else:
            raise(ValueError())
    
    def get_log(self, job_id):
        logfile = jobs_path / (job_id + '.log')
        if logfile.is_file():
            return logfile.read_text()
        else:
            return None
