import json, os, time, math
from zipfile import ZipFile
from io import StringIO
from collections import defaultdict, OrderedDict

def load(filename):
    """ Loads the file named *filename*, and instantiates one of the
        :class:`Data` subclasses to represent the data within that file. The
        filetypes
        .json, .bson and .yaml are supported and the filename
        has to end with one of those suffixes. The file is assumed to have a top
        level dictionary and if that dictionary has an *object_class* key,
        its value will specify which class to instantiate. Otherwise the
        superclass :class:`Data` is used. The remaning keys of the top level
        dictionary will become attributes on the returned data object. Also, the
        filename will be stored in the *_filename* attribute.        
    """
    if not os.path.exists(filename) or not os.path.isfile(filename):
        for suffix in ['.json', '.yaml', '.bson']:
            if os.path.exists(filename + suffix):
                filename += suffix
    with open(filename, 'r') as fd:
        if filename.endswith('.json'):
            data = json.load(fd)
        elif filename.endswith('.bson'):
            import bson
            data = bson.BSON(fd.read()).decode()
        elif filename.endswith('.yaml'):
            import yaml
            data = yaml.load(fd, Loader=yaml.CSafeLoader)
        else:
            raise NameError('Unknown file extention in "%s". Supported formats is .json, .bson and .yaml' % filename)

    if 'data_class' in data:
        cls = globals()[data['data_class']]
        del data['data_class']
    else:
        cls = Data
    data['_filename'] = os.path.abspath(filename)
    return cls(**data)
    
def ground_overlap(a, b):
    """ Matches *a* and *b* using the amount of overlap between their ground 
        polygones as match criteria and the distance between their world positions 
        as distance measure.
    """
    from shapely.geometry import Polygon
    r1, r2 = Polygon(a.ground_polygon), Polygon(b.ground_polygon)
    r = r1.intersection(r2)
    m = r.area / float(r1.area + r2.area - r.area)   
    
    x1, y1 = a.world_position
    x2, y2 = b.world_position
    d = math.sqrt( (x1 - x2)**2.0 + (y1 - y2)**2.0 )
    
    return m, d
ground_overlap.distance_unit = 'meters'

def bounding_box_overlap(a, b):
    """ Matches *a* and *b* using the amount of overlap between their bounding 
        boxes within all camera images as match criteria and the sum of the 
        distances between the centers of their bounding boxes in all camera 
        images as distance measure.
    """    
    overlap = total = distance = 0
    for i in range(len(a.bounding_boxes)):
        r1, r2 = a.bounding_box(i), b.bounding_box(i)
        r = r1.intersect(r2)
        overlap += r.area
        total += float(r1.area + r2.area - r.area)
        
        x1, y1 = r1.center
        x2, y2 = r2.center
        distance +=  math.sqrt( (x1 - x2)**2.0 + (y1 - y2)**2.0 )
    m = overlap / float(total)
    
    return m, distance
bounding_box_overlap.distance_unit = 'pixels'    

class Data(object):
    """ Common superclass of all the data representation classes. It takes a
        general set of keyword arguments that will be converted to attributes on
        the created object. In addition to those attributes it also have the
        following members:
    """
    _default_flow_style = None
    
    def __init__(self, **data):
        self.__dict__.update(data)
        self._setup()
        
    def _setup(self):
        pass

    def _load_child(self, filename):
        return load(self._abspath(filename))
        
    def _get_data(self):
        return {k: v for k, v in list(self.__dict__.items()) if not k.startswith('_')}
        
    def _abspath(self, filename):
        """ Convert a filename relative to the directory containing the current 
            file to an absolute path. The *filename* parameter can be either a 
            string/unicode or a list to be passed to os.path.join. If the 
            filename has no suffix the suffix of the current file will be appended.
        """
        if not isinstance(filename, str):
            filename = os.path.join(*filename)
        if hasattr(self, '_filename'):
            if '.' not in os.path.basename(filename):
                filename += self._filename[self._filename.rindex('.'):]
            return os.path.join(os.path.dirname(self._filename), filename)
        return filename

    @property            
    def json(self):
        """ The data represented by the object encoded as json. """
        data = self._get_data()
        data['data_class'] = self.__class__.__name__
        return json.dumps(data)
        
    @property            
    def yaml(self):
        """ The data represented by the object encoded as yaml. """ 
        import yaml
        data = self._get_data()
        data['data_class'] = self.__class__.__name__
        return yaml.dump(data, encoding='utf-8', Dumper=yaml.CSafeDumper, default_flow_style=self._default_flow_style)
        
    @property            
    def bson(self):
        """ The data represented by the object encoded as bson. """        
        import bson
        data = self._get_data()
        data['data_class'] = self.__class__.__name__
        return bson.BSON.encode(data)

    def save(self, filename=None):
        """ Saves the data represented by the object to disk. If the *filename*
            parameter is not given, the *self._filename* attribute will be
            used. This attribute is assigned when the object is loaded by
            :func:`load`. The filename has to end with one of the supported
            suffixes (currently .json, .bson and .yaml), which specifies the
            file format used.
        """
        if filename is None:
            filename = self._filename
        with open(filename, 'w') as fd:
            if filename.endswith('.json'):            
                fd.write(self.json + '\n')
            elif filename.endswith('.bson'):
                fd.write(self.bson + '\n')
            elif filename.endswith('.yaml'):
                fd.write(self.yaml + '\n')
            else:
                raise NameError('Unknown file extention in "%s". Supported formats is .json, .bson and .yaml' % filename)
                
    def make_avi(self, filename):
        print("Producing avi video for %s is not supported." % self.__class__.__name__)
        from imgpy.io import Mencoder
        from imgpy.image import Image
        avi = Mencoder(filename)
        frame = Image(128,8, 'B')
        frame[:,:] = 255
        for i in range(1):
            avi.view(frame)
            
class TsaiCamera(Data):
    """ Subclass of :class:`Data`
        representing a calibrated camera using the same model as the
        Tsai camera calibration system. It is constructed by loading
        a calibration file using :func:`load`, or by specifying all
        attributes as keyword arguments to the constructor. The attributes are
        *dx*, *dy*, *Cx*, *Cy*, *Sx*, *f*, *k*, *Tx*, *Ty*, *Tz*, *r1*, 
        *r2*, *r3*, *r4*, *r5*, *r6*, *r7*, *r8* and *r9*.
        For more information on the model and a description of the different
        attributes, see:
        
            http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
            
        The calibration is used to relate world coordinates expressed in meters
        with image coordinates expressed in pixels. The ground plane is
        typically assumed to be the plane *z=0*.
    """
    
    _default_flow_style = False

    
    def world_to_image(self, x, y=None, z=None):
        """ Projects the world coordinate *(x, y, z)* expressed in meters into 
            the image coordinate *(u, v)* expressed in pixels. It can be called
            in two equivalent ways:
                
                .. code-block:: python

                    u, v = cam.world_to_image(x, y, z)
                    
                    pkt = (x, y, z)
                    u, v = cam.world_to_image(pkt)
                
            The z coordinate can be omitted (in both case can) and will then
            assume its default value 0.0.
        """
        if y is None:
            assert z is None
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
        if z is None:
            z = 0.0
    
        # world to camera XYZ
        Xc = self.r1 * x + self.r2 * y + self.r3 * z + self.Tx
        Yc = self.r4 * x + self.r5 * y + self.r6 * z + self.Ty
        Zc = self.r7 * x + self.r8 * y + self.r9 * z + self.Tz
    
        # camera XYZ to undistrorted sensor
        Xu = self.f * Xc / Zc
        Yu = self.f * Yc / Zc
    
        #undistorted sensor to distorted sensor
        prev, l = 0.0, 1.0
        Xd, Yd = Xu, Yu
        maxiter = 20
        while abs(prev/l - 1.0) > 1e-15 and maxiter > 0:
            prev = l
            r = Xd ** 2 + Yd ** 2
            l = (1 + self.k * r)
            Xd, Yd = Xu/l, Yu/l
            maxiter -= 1    
        #distorted sensor to final image
        u = self.Sx * Xd / self.dx + self.Cx
        v = Yd / self.dy + self.Cy
        return u, v
        
    def image_to_world(self, xxx_todo_changeme, z=0.0):
        """ Projects the image coordinate *(u, v)*  expressed in pixel into
            a world coordinate *(x, y, z)* expressed in meters and returns it.
            Multiple world coordinates project into the same pixel. By default
            the world coordinate with *z = 0* is returned. The optional argument
            *z* can be used to choose a different one.
        """
        (u, v) = xxx_todo_changeme
        Xd = self.dx * (u - self.Cx) / self.Sx
        Yd = self.dy * (v - self.Cy)
        
        
        #convert from distorted sensor to undistorted sensor plane coordinates
        r = Xd ** 2 + Yd ** 2
        Xu = Xd * (1 + self.k * r)
        Yu = Yd * (1 + self.k * r)
        
        
        #calculate the corresponding Xw and Yw world coordinates
        cd = ((self.r1 * self.r8 - self.r2 * self.r7) * Yu + 
              (self.r5 * self.r7 - self.r4 * self.r8) * Xu - 
               self.f * self.r1 * self.r5 + self.f * self.r2 * self.r4)
        
        x = (((self.r2 * self.r9 - self.r3 * self.r8) * Yu +
              (self.r6 * self.r8 - self.r5 * self.r9) * Xu -
               self.f * self.r2 * self.r6 + self.f * self.r3 * self.r5) * z +
              (self.r2 * self.Tz - self.r8 * self.Tx) * Yu +
              (self.r8 * self.Ty - self.r5 * self.Tz) * Xu -
               self.f * self.r2 * self.Ty + self.f * self.r5 * self.Tx) / cd
        
        y = -(((self.r1 * self.r9 - self.r3 * self.r7) * Yu +
               (self.r6 * self.r7 - self.r4 * self.r9) * Xu -
                self.f * self.r1 * self.r6 + self.f * self.r3 * self.r4) * z +
               (self.r1 * self.Tz - self.r7 * self.Tx) * Yu +
               (self.r7 * self.Ty - self.r4 * self.Tz) * Xu -
                self.f * self.r1 * self.Ty + self.f * self.r4 * self.Tx) / cd
                
        return x, y, z
        
class Homography(Data):
    """ Subclass of :class:`Data` representing a Homography, *H*, relating the 
        camera image coordinates, *x*, to the ground plane z=0 coordinates *X* as
        X = H*x. 
    """
    _default_flow_style = False

    def _setup(self):
        self._inv1 = None        
        
    @property
    def numpy_matrix(self):
        from numpy import matrix
        return matrix([[self.h1, self.h2, self.h3],
                       [self.h4, self.h5, self.h6],
                       [self.h7, self.h8, self.h9]])
    
    def world_to_image(self, x, y=None, z=None):
        """ Projects the world coordinate *(x, y, 0)* expressed in meters into 
            the image coordinate *(u, v)* expressed in pixels. It can be called
            in two equivalent ways:
                
                .. code-block:: python

                    u, v = cam.world_to_image(x, y, 0)
                    
                    pkt = (x, y, 0)
                    u, v = cam.world_to_image(pkt)
                
            The z coordinate can be omitted but if it is specified it has to be 0.
        """
        if y is None:
            assert z is None
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
        assert z is None or z == 0.0
        
        if self._inv1 is None:
            (self._inv1, self._inv2, self._inv3,
            self._inv4, self._inv5, self._inv6,
            self._inv7, self._inv8, self._inv9) = self.numpy_matrix.I.flat
        a = self._inv1 * x + self._inv2 * y + self._inv3
        b = self._inv4 * x + self._inv5 * y + self._inv6
        c = self._inv7 * x + self._inv8 * y + self._inv9
        return a/c, b/c
            
        
    def image_to_world(self, xxx_todo_changeme1, z=0.0):
        """ Projects the image coordinate *(u, v)*  expressed in pixel into
            a world coordinate *(x, y, 0)* expressed in meters and returns it.
            Multiple world coordinates project into the same pixel. The world 
            coordinate with *z = 0* is returned. The optional argument
            *z* must be 0 if specified.
        """        
        (u, v) = xxx_todo_changeme1
        assert z == 0.0
        a = self.h1 * u + self.h2 * v + self.h3
        b = self.h4 * u + self.h5 * v + self.h6
        c = self.h7 * u + self.h8 * v + self.h9
        return a/c, b/c, 0.0
        
    
class ZipVideo(Data):
    """ Subclass of :class:`Data` representing a video stored on disk as a zip
        file containing jpeg images. It is constructed by loading a video file
        using :func:`load` or by specifying its attributes as keyword arguments
        to the constructor. Some of the attributes are calculated if not given.
        The name of each jpeg frame in the archive should be it's timestamp in
        the canonical format YYYYMMDD-hhmmss.iii.jpg, where YYYY is the year, MM
        the month, DD then day, hh the hour mm the minutes ss the seconds and
        iii the miliseconds.
        
        Typical usage is to iterate over the frames of the video in a for loop.
        The frames are returned as :class:`Frame` objects.
        
            .. code-block:: python
            
                video = load('Video/Camera_1.yaml')
                
                for frame in video:
                    image = frame.load_pil()
                    # ...
                    
                first_frame = video[0]
                    
        To only iterate over part of the video, slices can be used. To iterate
        over several parts in a single loop, use *+* to concatinate the sliced
        views:

            .. code-block:: python
            
                for frame in video[100:200]:
                    image = frame.load_pil()
                    # ...

                for frame in video[100:200] + video[300:400]:
                    image = frame.load_pil()
                    # ...
                    
        The attributes stored are:
         
            **video_zip_file**
                The filename of the zip file containing the frames as jpeg
                images.
            **time_offset**
                An time offset relating the timestamps to a master clock 
                common for all cameras. Defaults to 0.0.
            **time_scale**
                A time rescale factor relating the timestamps to a master 
                clock common for all cameras. Defaults to 1.0.
            **calibration_file**
                The filename of a file storing a :class:`TsaiCamera` object with
                the calibration of the camera used to record the video.
            **scene_file**
                The filename of a file storing a :class:`Scene` storing metadata
                about the scene.
                
        In addition to the stored attributes, the loaded object also has the
        following members:

    """
    def _setup(self):
        zipfile = self._abspath(self.video_zip_file)
        self._video_zip = ZipFile(zipfile, 'a', allowZip64=True)
        self._frame_name = [n for n in self._video_zip.namelist() if n.endswith('.jpg')]
        self._frame_name.sort()
        self._frame_index = None
        if not hasattr(self, 'time_offset'):
            self.time_offset = 0.0
        if not hasattr(self, 'time_scale'):
            self.time_scale = 1.0
        self._calibration = None
        self._scene = None
        self._frame_times = None
            
    def add_frame(self, jpeg, timestamp):
        """ Add the frame *jpeg* to the video stream with timestamp *timestamp* and 
            append it to the zip file. The *jpeg* should be a string containing a 
            jpeg encoded image.
        """
        t = time.localtime(timestamp)
        ms = int((timestamp - int(timestamp)) * 1000)
        timestring = time.strftime('%Y%m%d-%H%M%S', t) + '.%.3d' % ms
        name = timestring + '.jpg'
        if name in self._frame_name:
            raise AttributeError('Frame %s already exists.' % timestring)
        self._video_zip.writestr(name, jpeg)
        self._frame_name.append(name)
        self._frame_name.sort()
        self._frame_index = None
        self._frame_times = None        
            
    @property
    def frame_count(self):
        """ The number of frames in the video. """
        return len(self)
        
    @property
    def size(self):
        """ The width and heigh of the video in pixels. """
        return self[0].load_pil().size
        
    @property
    def fps(self):
        """ The mean number of frames per second. Due to framedrops the
            framerate might vary slightly over video as specified by the
            *timestamps* attribute. """
        return len(self) / (self[-1].timestamp - self[0].timestamp)

    def timestamp(self, i):
        """ Returns the timestamp in seconds of the frame with index *i*. """
        d, ms, _ = self._frame_name[i].split('.')
        t = list(time.strptime(d, '%Y%m%d-%H%M%S'))
        t[0] = max(1900, t[0])
        ts = time.mktime(t)
        return self.time_scale * (ts + float('0.' + ms)) + self.time_offset

    def timestring(self, i):
        """ Resturns the timestring identifier of the frame with inedx *i*. That
            is the timestamp of the frame as a string in the canonical form
            described above. """
        n = self._frame_name[i]
        return n[:n.rindex('.')]
        
    @property
    def calibration(self):
        """ A :class:`TsaiCamera` object with the calibration of the camera used
            to record the video. It is loaded from the file *calibration_file*.
        """
        if self._calibration is None:
            self._calibration = self._load_child(self.calibration_file)
        return self._calibration
        
    @property
    def scene(self):
        """ A :class:`Scene` object with metadata about the scene. It is loaded 
            from the file *scene_file*.
        """
        if self._scene is None:
            self._scene = self._load_child(self.scene_file)
        return self._scene

    def __len__(self):
        """ The number of frames in the video. """
        return len(self._frame_name)
        
    def __getitem__(self, i):
        """ If *i* is int, returns frame number *i*. If *i* is a slice, retuns a
            sliced view. If *i* is str, return the frame with that exact
            timestring. """
        if isinstance(i, slice):
            return SliceView(self, i)
        if isinstance(i, str):
            if self._frame_index is None:
                self._frame_index = {self.timestring(i): i for i in range(len(self))}
            i = self._frame_index[i]
        if i < 0:
            i += len(self)
        if 0 <= i < self.frame_count:
            return Frame(self, i)
        raise IndexError

    def frame_at_time(self, ts):
        """ Returns the frame closest in time to *ts*, which is a float
            timestamp. """
        if self._frame_times is None:
            self._frame_times = [f.timestamp for f in self]
        i, d = 0, float('Inf')
        while True:
            oldd = d
            d = (ts - self._frame_times[i]) * self.fps
            if abs(oldd) <= abs(d):
                break
            oldi = i
            i += int(d)
            if i == oldi:
                i += int(math.copysign(1.0, d))
            i = max(min(i, len(self)-1), 0)
        return self[oldi]
        
    def show(self):
        """ Plays the video in a pygame window. """
        import pygame
        pygame.init()
        try:
            screen = pygame.display.set_mode(self.size)
            for frame in self:
                image = frame.load_pygame()
                screen.blit(image, [0,0])
                pygame.display.flip()
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT:
                        raise Done
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:                    
                        raise Done
        except Done:
            pass
        finally:
            pygame.quit()
            
    def make_avi(self, filename):
        """ Converts the video to an .avi file. """
        from imgpy.io import Mencoder
        fps = len(self) / (self.timestamp(-1) - self.timestamp(0))
        avi = Mencoder(filename, options=['-fps', str(fps)])
        for frame in self:
            avi.view(frame.load_imgpy())

    def as_synced(self, filename=None):
        """ Return this video as a :class:`SyncedVideos` object containing a single 
            camera with all the frames from this video.
        """
        params = {
            'master_timestamps': [f.timestamp for f in self],
            'video_timestrings': [[f.timestring] for f in self]
        }
        if filename:
            video_file = os.path.relpath(self._filename, os.path.dirname(filename))
            video_file = video_file[:video_file.rindex('.')]
            synced = SyncedVideos(_filename=filename, video_files=[video_file], **params)
        else:
            video_file = self._filename[:self._filename.rindex('.')]
            synced = SyncedVideos(video_files=[video_file], **params)
        return synced
        



 
class SliceView(object):
    def __init__(self, parent, index):
        self.parent = parent
        if isinstance(index, slice):
            self.range = range(*index.indices(len(parent)))
        else:
            self.range = index
        
    def __len__(self):
        return len(self.range)
        
    def __getitem__(self, i):
        if isinstance(i, slice):
            return SliceView(self, i)
        else:
            return self.parent[self.range[i]]
            
    def __add__(self, other):
        assert self.parent is other.parent
        return SliceView(self.parent, list(self.range) + list(other.range))

class Frame(object):
    """ Represents a single frame within an :class:`ZipVideo` object. It has
        the attributes:
        
            **video**
                The :class:`ZipVideo` object this frame belongs to
            **index**
                The index of the frame, i.e. the object
                *frm* would refer to the same frame as *frm.video[frm.index]*.
    """
    def __init__(self, video, index):
        self.video = video
        self.index = index
        
    @property
    def timestamp(self):
        """ The timestamp in seconds of the frame as a float."""
        return self.video.timestamp(self.index)

    @property
    def timestring(self):
        """ The timestamp as a string in its canonical format. """
        return self.video.timestring(self.index)
        
    def load_jpeg(self):
        """ Loads and returns the frame as compressed jpeg data. """
        return self.video._video_zip.open(self.video._frame_name[self.index]).read()
        
    def load_pil(self):
        """ Loads and returns the frame as a PIL Image. """
        import PIL.Image
        r = PIL.Image.open(StringIO(self.load_jpeg()))
        return r
        
    def load_pygame(self):
        """ Loads and returns the frame as a PyGame Surface. """
        import pygame.image
        return pygame.image.load(StringIO(self.load_jpeg()))
        
    def load_imgpy(self):
        """ Loads and returns the frame as an ImgPy RGBImage. """
        from imgpy.io import imread
        img = imread(StringIO(self.load_jpeg()))
        try:
            img.cal = self.video.calibration
        except AttributeError:
            img.cal = None
        return img
        
    def load_numpy(self):
        """ Loads and returns the frame as a NumPy ndarray. """
        import numpy
        img = self.load_pil()
        return numpy.ndarray((img.size[1], img.size[0], 3), 'B', img.tostring())
                
class SyncedVideos(Data):
    """ Represents a set of synchronized :class:`ZipVideo` streams.
        It consists of a stream of :class:`SyncedFrame` frames that each
        represents one frame from each video stream that is exposed as close as
        possible in time. It is constructed by resampling the original videos
        based on their timestamps, using a common master clock with constant
        frame-rate. The typical usage would be to iterate over those
        :class:`SyncedFrame` to for example display the videos in sync:
        
            .. code-block:: python
                
                synced = load('Video/SyncedVideo1234_10fps.yaml')
                for frames in synced:
                    images = [frm.load_pygame() for frm in frames]
                    # Show the images
                    
        The attributes stored are:
        
            **master_timestamps**
                A list of master timestamps that represents the global clock
                that the separate cameras are synchronized to. These are
                typically synthetically generated with constant frame-rate.
                They are generated from *fps* if not specified.
            **fps**
                The number of frames per second in the synchronized stream It is
                calculated from *master_timestamps* if not specified.
            **video_files**
                A list of the video files synchronized.
            **video_timestrings**
                A list with one item for each synced frame consisting of a list
                of one timestring for each video stream specifying which frame
                in that video that belongs to the synced frame. It is generated 
                by finding the timestamp in each video closest to each timestamp
                in master_timestamps if not specified.
    """
    def _setup(self):
        if not hasattr(self, 'fps'):
            self.fps = (len(self) - 1) / (self.master_timestamps[-1] - self.master_timestamps[0])
        self._videos = [self._load_child(fn) for fn in self.video_files]
        if hasattr(self, 'master_timestamps') or hasattr(self, 'video_timestrings'):
            assert len(self.master_timestamps) == len(self.video_timestrings)
        else:
            t0 = max(v[0].timestamp for v in self.videos)
            t1 = min(v[-1].timestamp for v in self.videos)
            n = int((t1 - t0) * self.fps + 1.5)
            self.master_timestamps = [t0 + i / float(self.fps) for i in range(n)]
            self.video_timestrings = [[v.frame_at_time(t).timestring for v in self.videos]
                                      for t in self.master_timestamps]
        self._frame_times = None
    
    @property
    def videos(self):
        """ A list of :class:`ZipVideo` objects loaded from *video_files*. """
        return self._videos

    def __len__(self):
        """ The number of frames in the synchronized stream. """
        return len(self.master_timestamps)

    def __getitem__(self, i):
        """ If *i* is a slice, a slice view is returned. If *i* is an int, a
            :class:`SyncedFrame` referring to the frame with index *i* in the
            synced stream.
        """
        if isinstance(i, slice):
            return SliceView(self, i)
        timestrings = self.video_timestrings[i]
        return SyncedFrame(self, i, [cam[timestrings[c]] for c, cam in enumerate(self.videos)])
        
    def timestamp_index(self, ts):
        """ Returns the frame index of the frame with the exact timestamp *ts*,
            which is a float timestamp. """
        if self._frame_times is None:
            self._frame_times = {ts: i for i, ts in enumerate(self.master_timestamps)}
        return self._frame_times[ts]
        
    def frame_at_time(self, ts):
        """ Returns the frame with the exact timestamp *ts*, which is a float
            timestamp. """
        return self[self.timestamp_index(ts)]
        
    def show(self):
        """ Plays the videos side by side in sync in a pygame window. """
        import pygame
        width = sum(v.size[0] for v in self.videos)
        height = max(v.size[1] for v in self.videos)
        pygame.init()
        try:
            screen = pygame.display.set_mode([width, height])
            for frames in self:
                x = 0
                for frm in frames:
                    image = frm.load_pygame()
                    screen.blit(image, [x, 0])
                    x += image.get_width()
                pygame.display.flip()
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT:
                        raise Done
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:                    
                        raise Done
        except Done:
            pass
        finally:
            pygame.quit()
            
    def make_avi(self, filename):
        """ Produces an .avi file with the videos side py side in sync."""
        from imgpy.io import Mencoder
        from imgpy.image import hcat
        avi = Mencoder(filename, options=['-fps', str(self.fps)])
        for frames in self:
            img = hcat(*[frm.load_imgpy() for frm in frames])
            avi.view(img)

        

class SyncedFrame(object):
    """ Represents a set of video frames from different videos that were
        exposed at approximately the same time. It has the attributes:
        
            **index**
                The index of the represented frame in the synced stream.
    """
    def __init__(self, synced, index, frames):
        self.synced = synced
        self.index = index
        self._frames = frames

    def __getitem__(self, i):
        """ Returns the frame from the video with index *i*. """
        return self._frames[i]
        
    def __len__(self):
        """ Returns the number of videos in the synced stream. """
        return len(self._frames)
        
    @property
    def timestamp(self):
        """ The master timestamps of this frame. """
        return self.synced.master_timestamps[self.index]
        

class TrackSet(Data):
    """ Subclass of :class:`Data` representing an ordered set of :class:`Track`
        objects. It can be used to represent manually annotated ground truth, in
        which case it is typically loaded using :func:`load`. It is used to
        access :class:`Track` and :class:`State` objects. Either organized
        track by track:

            .. code-block:: python
            
                track_set = load("Trajectories/RoadUsers.yaml")                
                first_track = track_set[0]
                
                for track in track_set:
                    for state in track:
                        # ...

        or organized by frame:
        
            .. code-block:: python
            
                track_set = load("Trajectories/RoadUsers.yaml")
                
                for synced_frames in track_set.synced:
                    states = track_set.get_frame_detections(synced_frames.index)
                    for s in states:
                        # ...
                        
        It can also be used to store the result of a tracking algorithm and
        compare it with the ground truth:
        
            .. code-block:: python
            
                ground_truth = load("Trajectories/RoadUsers.yaml")
                track_set = TrackSet()
                
                # Find a car visible from frame first to frame last
                track = Track(type="Car", width=2.0, length=4.0, height=1.2)
                track_set.append(track)
                for synced_frames in ground_truth.synced[first:last+1]:
                    # Find the location of the car in frame as x, y in 
                    # world coordinates (m)
                    track.append(State(world_position=(x, y), 
                                       frame=frame.index))
                
                print ground_truth.compare(track_set)
                
                        
        It has the following attributes:
        
            **tracks**
                A list of :class:`Track` objects
            **synced_file**
                The filename of a file storing a :class:`SyncedVideos` object with
                the videos from which the tracks originates.
            

    """
    def _setup(self):
        if hasattr(self, 'tracks'):
            self.tracks = [Track(**t) for t in self.tracks]
        else:
            self.tracks = []
        self._synced = None
        self._present = defaultdict(set)
        for i, t in enumerate(self.tracks):
            t._index = i
            t._trackset = self
            for d in t:
                self._present[d.frame].add(t)
                if hasattr(self, 'synced_file'):
                    d._calc_missing_data(self.synced)
                
    def append(self, track):
        """ Append the :class:`Track` object *track* to the set of tracks."""
        self.tracks.append(track)
        assert track._trackset is None
        track._trackset = self
        for d in track:
            self._present[d.frame].add(track)
            if hasattr(self, 'synced_file'):
                d._calc_missing_data(self.synced)
        
                
    def _get_data(self):
        data = Data._get_data(self)
        data['tracks'] = [t._get_data() for t in self]
        return data

    def __len__(self):
        """ The number of tracks in the set. """
        return len(self.tracks)
        
    def __iter__(self):
        """ Iterate over all the tracks. """
        return iter(self.tracks)
        
    def __getitem__(self, idx):
        """ Get the track with index *idx*. """
        return self.tracks[idx]
            
    @property
    def synced(self):
        """ A :class:`SyncedVideos` object with the videos from which the tracks
            originates loaded from *synced_file*.
        """
        if self._synced is None:
            self._synced = self._load_child(self.synced_file)        
        return self._synced
        
    def get_frame_detections(self, frame):
        """ Return the :class:`State` objects with frame index *frame* from
            all the :class:`State` objects within all tracks in the set.
        """
        return set(t[frame] for t in self._present[frame])
    
    def compare(self, other, match_threshold=1.0/3.0, match_function=ground_overlap, bytype=False):
        """ Compare two :class:`TrackSet` objects. Typically one representing
            the ground truth and one representing the result of an
            automated tracking algorithm to be tested. It returns a
            :class:`Result` object that can be used like this:
            
                .. code-block:: python
                
                    result = ground_truth.compare(tracks)
                    print result
                    result.plot_dists()
                    
            By default, two :class:`State` objects are considered to match if 
            their ground rectangles overlap with an larger amount than 
            *match_threshold*. The assignment between tracks that maximizes the 
            total number of assigned states are used. Assignments whose states 
            match for less than 33% of their combined set of states are considered 
            partial and discarded .
            
            The *match_function* parameter can be used to change the matching 
            criteria. In addition to the default function, 
            :func:`ground_overlap` there is also :func:`bounding_box_overlap`
            that considers the overlap between bounding boxes instead of 
            ground footprints. These functions return a tuple with two values.
            The first is the match score which is compared to *match_threshold* 
            and the second is a distance measure which is used by 
            :func:`Result.plot_dists`.

        """
        try:
            if self.synced._filename != other.synced._filename:
                print('Warning: you are comparing tracks  produced from')
                print('            ', self.synced._filename)
                print('         with tracks produced from')
                print('            ', other.synced._filename)
                print()
        except AttributeError:
            pass

        def cost(a, b):
            return -len(a.matching_states(b, match_threshold, match_function))
        typed_res = {}
        res = Result(match_function.distance_unit)      
        for t1, t2 in hungarian(self, other, cost):
            if bytype:
                if t1 and t1.type not in typed_res:
                    typed_res[t1.type] = Result(match_function.distance_unit)       
                res = typed_res[t1.type] if t1 else typed_res[t2.type]
            if t1 is None:
                res.extra_tracks += 1
                res.extra_states += len(t2)
            elif t2 is None:
                res.missed_tracks += 1
                res.missed_states += len(t1)
            else:
                dists = t1.matching_states(t2, match_threshold, match_function)
                overlap = len(dists) / float(len(t1) + len(t2) - len(dists))
                if overlap > 1.0/3.0: # and (not bytype or t1.type == t2.type):            
                    res.matched_tracks += 1
                    res.matched_states += len(dists)
                    res.missed_states += len(t1) - len(dists)
                    res.extra_states  += len(t2) - len(dists)
                    res.dists.extend(dists)
                else:
                    if bytype:
                        typed_res[t2.type].extra_tracks += 1
                        typed_res[t2.type].extra_states += len(t2)
                        typed_res[t1.type].missed_tracks += 1
                        typed_res[t1.type].missed_states += len(t1)
                    else:
                        res.extra_tracks += 1
                        res.extra_states += len(t2)
                        res.missed_tracks += 1
                        res.missed_states += len(t1)
                    
        if bytype:
            return typed_res
            
        assert res.matched_tracks + res.missed_tracks == len(self)
        assert res.matched_tracks + res.extra_tracks == len(other)
        assert res.matched_states + res.missed_states == sum(len(t) for t in self)
        assert res.matched_states + res.extra_states == sum(len(t) for t in other)
        return res
        
    def show(self):
        """ Plays the :class:`SyncedVideos` that these tracks belongs to in a
            pygame window with the tracks annotated on top of the video.
        """
        import pygame
        width = sum(v.size[0] for v in self.synced.videos)
        height = max(v.size[1] for v in self.synced.videos)
        first = min(t.first_frame for t in self)
        last = max(t.last_frame for t in self)

        pygame.init()
        try:
            screen = pygame.display.set_mode([width, height])
            font = pygame.font.Font(None, 30)
            t = time.time()
            for frames in self.synced[first:last+1]:
                offset = 0
                for cam, frm in enumerate(frames):
                    image = frm.load_pygame()
                    screen.blit(image, [offset, 0])
                    for det in self.get_frame_detections(frames.index):
                        if det.camera_positions[cam]:
                            x, y = det.camera_positions[cam]
                            # print x, y
                            text = font.render(str(det.track.index), 1, (255,0,0))
                            w, h = text.get_size()
                            if 0 <= x - w/2.0 < image.get_width():
                                screen.blit(text, (x - w/2.0 + offset, y - h/2.0)) 
                                box = det.bounding_box(cam).translate(offset, 0)
                                pygame.draw.rect(screen, (0,0,255), box.to_pygame(), 2)
                                try:
                                    ground = [self.synced.videos[cam].calibration.world_to_image(point)
                                              for point in det.ground_polygon]
                                    roof = [self.synced.videos[cam].calibration.world_to_image(point)
                                              for point in det.roof_polygon]
                                except AttributeError:
                                    pass
                                else:
                                    pygame.draw.lines(screen, (255,0,0), True, [(x + offset, y) for x, y in ground])
                                    pygame.draw.lines(screen, (255,0,0), True, [(x + offset, y) for x, y in roof])
                                    for (x1, y1), (x2, y2) in zip(ground, roof):
                                        pygame.draw.line(screen, (255,0,0), (x1+offset, y1), (x2+offset, y2))

                                    
                    offset += image.get_width()
                time.sleep(max(t - time.time(), 0))
                pygame.display.flip()
                t += 1.0 / self.synced.fps                       
                for event in pygame.event.get(): 
                    if event.type == pygame.QUIT:
                        raise Done
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:                    
                        raise Done
        except Done:
            pass
        finally:
            pygame.quit()
    
    def make_avi(self, filename):
        """ Produces an .avi file with the videos side by side in sync and the 
            tracks annotated on top of the video.
        """
        from imgpy.io import Mencoder
        from imgpy.image import hcat
        from imgpy.draw import draw_text, draw
        first = min(t.first_frame for t in self)
        last = max(t.last_frame for t in self)
        avi = Mencoder(filename, options=['-fps', str(self.synced.fps)])
        
        for frames in self.synced:
            images = [frm.load_imgpy() for frm in frames]
            img = hcat(*images)
            offset = 0
            for cam, frm in enumerate(frames):
                for det in self.get_frame_detections(frames.index):
                    if det.camera_positions[cam]:
                        x, y = det.camera_positions[cam]
                        if 0 <= x < images[cam].width:
                            draw_text(img, (x + offset, y), str(det.track.index))
                            box = det.bounding_box(cam).translate(offset, 0)
                            draw(img, [(box.x0, box.y0), (box.x0, box.y1), 
                                       (box.x1, box.y1), (box.x1, box.y0),
                                       (box.x0, box.y0)], 'r-')
                offset += images[cam].width

            avi.view(img)

    def resample(self, synced):
        assert all(os.path.samefile(f1, f2)
                   for f1, f2 in zip([self.synced._abspath(f) for f in self.synced.video_files],
                                     [synced._abspath(f) for f in synced.video_files]))
        # XXX: Warn if there are more than one camera?
        new = TrackSet(synced_file=synced._filename) # FIXME: Copy data
        frame_map = {tuple(ts): i for i, ts in enumerate(synced.video_timestrings)}
        for track in self.tracks:
            new_track = Track(type=track.type) # FIXME: Copy data
            for state in track:
                try:
                    frame = frame_map[tuple(self.synced.video_timestrings[state.frame])]
                except KeyError:
                    pass
                else:
                    new_track.append(State(frame=frame, bounding_boxes=state.bounding_boxes)) # FIXME: Copy data
            new.append(new_track)
        return new

         
class Result(object):
    """ Represent the result of comparing a set of tracks, that each
        consists of a sequence of states, with their ground truth. It is
        produced using :func:`TrackSet.compare`. It has the following
        properties:
        
            **matched_tracks**
                The number of tracks that was detected.
            **missed_tracks**
                The number of tracks that was not detected.
            **extra_tracks**
                The number of additional false tracks reported.            
            **matched_states**
                The number of states that was detected.
            **missed_states**
                The number of states that was not detected.            
            **extra_states**
                The number of additional false states reported.            
            
    """
    def __init__(self, dists_unit):
        self.matched_tracks = self.missed_tracks = self.extra_tracks = 0
        self.matched_states = self.missed_states = self.extra_states = 0
        self.dists_unit = dists_unit
        self.dists = []
        
    @property
    def true_tracks(self):
        """ The total number of true tracks """
        return self.matched_tracks + self.missed_tracks
                
    @property
    def true_states(self):
        """ The total number of true states """
        return self.matched_states + self.missed_states
        
    def __repr__(self):
        """ A string representing the result suitable for printing. """
        s = ""
        mp = self.matched_tracks * 100.0 / self.true_tracks
        s += "Found %d (%.1f %%) of %d tracks.\n" % (self.matched_tracks, mp, self.true_tracks)
        mp = self.missed_tracks * 100.0 / self.true_tracks
        s += "Missed %d (%.1f %%) of %d tracks.\n" % (self.missed_tracks, mp, self.true_tracks)
        mp = self.extra_tracks * 100.0 / self.true_tracks
        s += "Found %d (%.1f %%) extra false tracks among %d tracks.\n\n" % (self.extra_tracks, mp, self.true_tracks)
        
        mp = self.matched_states * 100.0 / self.true_states
        s += "Found %d (%.1f %%) of %d states.\n" % (self.matched_states, mp, self.true_states)
        mp = self.missed_states * 100.0 / self.true_states
        s += "Missed %d (%.1f %%) of %d states.\n" % (self.missed_states, mp, self.true_states)
        mp = self.extra_states * 100.0 / self.true_states
        s += "Found %d (%.1f %%) extra false states among %d states." % (self.extra_states, mp, self.true_states)
        return s
        
    def plot_dists(self):
        """ Produces and shows a plot describing the precision of the tested
            system. It shows for each distance, *x*, in meters (on the x-axis)
            the amount of states detected within that distance of their ground
            truth. Only matched states of matched tracks are considered. 
        """
        from pylab import plot, show, xlabel, ylabel
        self.dists.sort()
        h = [float(i)/self.true_states for i in range(len(self.dists))]
        plot(self.dists, h)
        xlabel('Maximum distance from ground truth (%s)' % self.dists_unit)
        ylabel('Amount of states')
        show()
        
    @staticmethod
    def plot_multi_dists(results, style='-', show=True):
        do_show = show
        from pylab import plot, show, xlabel, ylabel, legend
        colors = 'bgrcmyk'
        for tidx, (t, r) in enumerate(results.items()):
            r.dists.sort()
            h = [float(i)/r.true_states for i in range(len(r.dists))]
            plot(r.dists, h, colors[tidx] + style, label=t)
        if do_show:
            xlabel('Maximum distance from ground truth (%s)' % 
                   list(results.values())[0].dists_unit)
            ylabel('Amount of states')
            legend(loc='best')
            show()
        
        
        
class Track(Data):
    """ Subclass of :class:`Data` representing an object track in form of an
        ordered set of :class:`State` objects that repesents the state of the
        object in each frame. It has the  following attributes:
        
            **states**
                A list of :class:`State` objects representing the position
                of the object in each frame it is visible.
            **width**
                The width of the object in meters.
            **length**
                The length of the object in meters.
            **height**
                The height of the object in meters.
            **type**
                A string specifying the type of object ("Car", "Bus",
                "Pedestrian", "Bicycle", ...)
    """

    def _setup(self):
        states = OrderedDict()
        if hasattr(self, 'states'):
            for d in self.states:
                # if d=='camera_positions':
                #     import pdb; pdb.set_trace()
                det = State(**d)
                det._parent = self
                states[d['frame']] = det
        self.states = states
        self._frames = [d.frame for d in self]
        self._trackset = None
        
    def append(self, det):
        """ Append the :class:`State` object *det* to the
            set of states.
        """
        assert det._parent is None
        det._parent = self
        if self._trackset is not None:
            det._calc_missing_data(self._trackset.synced)
            self._trackset._present[det.frame].add(self)
        self._frames.append(det.frame)
        self.states[det.frame] = det
        
    def _get_data(self):
        data = Data._get_data(self)
        data['states'] = [d._get_data() for d in self]
        return data

    def __len__(self):
        """ The number of :class:`State` objects. """
        return len(self.states)
        
    def __iter__(self):
        """ Iterate over the :class:`State` objects. """
        return iter(self.states.values())
        
    def __getitem__(self, frame):
        """ Get the :class:`State` object representing the position for
            frame with index *frame*.
        """
        return self.states[frame]
        
    @property
    def first_frame(self):
        """ The index of the first frame of the track. """
        return self._frames[0]

    @property
    def last_frame(self):
        """ The index of the last frame of the track. """
        return self._frames[-1]
        
    @property
    def index(self):
        """ The index of this track within the :class:`TrackSet` it belongs to.
        """
        return self._index
        
    def interpolate(self):
        "XXX"
                
    def matching_states(self, other, match_threshold, match_function):
        """ Return the set of :class:`State` objects in *self* that matches a
            :class:`State` object in *other*. Two states are considered to
            match if their ground rectangles overlap with more than 50%.
        """
        f0 = max(self.first_frame, other.first_frame)
        f1 = min(self.last_frame, other.last_frame)
        dists = []
        for f in range(f0, f1+1):
            m, d = match_function(self[f], other[f])
            if  m > match_threshold:
                dists.append(d)
        return dists
            

class Rect(object):
    """ Representation of a rectangle """
    
    def __init__(self, xxx_todo_changeme2, xxx_todo_changeme3):
        """ Constructs a new with upper covering (*x*, *y*) for *x* = *x0*
            ... *x1* and *y* = *y0* .. *y1*.
        """
        (x0, x1) = xxx_todo_changeme2
        (y0, y1) = xxx_todo_changeme3
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def intersect(self, r):
        """ Returns a new :class:`Rect` rectangle representing  the intersection
            between *self* and *r*.
        """
        return Rect([max(self.x0, r.x0), min(self.x1, r.x1)],
                    [max(self.y0, r.y0), min(self.y1, r.y1)])

    @property
    def area(self):
        """ The area of the rectangle. """
        dx = self.width
        dy = self.height
        if dx <= 0 or dy <= 0:
            return 0
        else:
            return dx*dy
            
    @property
    def left(self):
        """ The left border of the rectangle """
        return self.x0
        
    @property
    def right(self):
        """ The right border of the rectangle """
        return self.x1
        
    @property
    def top(self):
        """ The top border of the rectangle """
        return self.y0
        
    @property
    def bottom(self):
        """ The bottom border of the rectangle """
        return self.y1
        
    @property
    def width(self):
        """ The width of the rectangle """
        return self.x1 - self.x0 + 1
            
    @property
    def height(self):
        """ The height of the rectangle """
        return self.y1 - self.y0 + 1
        
    @property
    def center(self):
        """ The center of the rectangle """
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def to_pygame(self):
        """ Returns a pygmae.Rect representation of the rectangle """
        import pygame
        return pygame.Rect(self.left, self.top, self.width, self.height)
        
    def translate(self, dx, dy):
        """ Returns a new Rect object that is translated (dx, dy) as compared
            to this rectangle, which is not changed.
        """
        return Rect((self.x0 + dx, self.x1 + dx), (self.y0 + dy, self.y1 + dy))

            
class State(Data):
    """ Subclass of :class:`Data` representing the position and orientation of
        an object in a single frame. It has the  following attributes:
        
            **frame**
                The index of the frame within the :class:`SyncedVideos` stream 
                this states refers to. It is calculated from *master_timestamp*
                if not specified.
            **master_timestamp**
                The master timestamp of the frame within the :class:`SyncedVideos` 
                stream this states refers to. It is calculated from *frame* if not
                specified.
            **world_position**
                The center position of the object on the ground plane (z=0) in
                world coordinates (meters).
            **world_direction**
                A manual annotated forward direction of the object within the 
                ground plane (z=0) expressed as a unit vector.
            **camera_positions**
                A list of pixel positions specifying the location of the object
                within each camera of the :class:`SyncedVideos` of the
                :class:`TrackSet` of the :class:`Track` that this state belongs
                to. If it is not specified, it is calculated from the
                *world_position* using the :class:`TsaiCamera` calibration of
                each video.
            **bounding_boxes**
                A list of bounding boxes in the format:
                                    
                        [(left, right), (top, bottom)]
                        
                It specifies the location of the object 
                within each camera of the :class:`SyncedVideos` of the
                :class:`TrackSet` of the :class:`Track` that this state belongs
                to. If it is not specified, it is calculated from the
                *world_position* using the :class:`TsaiCamera` calibration of
                each video and the *width*, *heigh* and *length* of the 
                :class:`Track` if this State belongs to a :class:`Track`.
    """
    _parent = None
        
    @property
    def track(self):
        """ The :class:`Track` this state belongs to if applicable. """
        assert isinstance(self._parent, Track)
        return self._parent
        
    @property
    def stateset(self):
        """ The :class:`StateSet` this state belongs to if applicable. """
        assert isinstance(self._parent, StateSet)
        return self._parent
        
    def _calc_missing_data(self, synced):
        if not hasattr(self, 'world_position'):
            if not hasattr(self, 'camera_positions'):
                self.camera_positions = [Rect(*bb).center for bb in self.bounding_boxes]
            pos = [v.calibration.image_to_world(p) 
                   for v, p in zip(synced.videos, self.camera_positions)]
            x = sum(p[0] for p in pos) / float(len(pos))
            y = sum(p[1] for p in pos) / float(len(pos))
            self.world_position = (x, y)
        if not hasattr(self, 'camera_positions'):
            self.camera_positions = [v.calibration.world_to_image(self.world_position)
                                     for v in synced.videos]
        if not hasattr(self, 'master_timestamp'):
            try:
                self.master_timestamp = synced.master_timestamps[self.frame]
            except IndexError:
                self.master_timestamp = None
        if not hasattr(self, 'frame'):
            self.frame = synced.timestamp_index(self.master_timestamp)
        if not hasattr(self, 'bounding_boxes') and isinstance(self._parent, Track):
            self.bounding_boxes = []
            for cam in synced.videos:
                left = top = float('Inf')
                right = bottom = float('-Inf')
                for point in self.ground_polygon + self.roof_polygon:
                    u, v = cam.calibration.world_to_image(point)
                    left = min(left, u)
                    top = min(top, v)
                    right = max(right, u)
                    bottom = max(bottom, v)
                self.bounding_boxes.append([(left, right), (top, bottom)])
            
    @property
    def ground_polygon(self):
        return self.polygon(0)

    @property
    def roof_polygon(self):
        return self.polygon(-self.track.height)

    def polygon(self, z):
        poly = []
        x, y = self.world_position  
        if hasattr(self, 'world_direction'):
            vx, vy = self.world_direction
        else:
            vx, vy = 1, 0
        for ort in (-self.track.width/2.0, self.track.width/2.0):
            for fwd in (-self.track.length/2.0, self.track.length/2.0):
                dx = fwd * vx - ort * vy
                dy = fwd * vy + ort * vx
                poly.append([x+dx, y+dy, z])
        return [poly[0], poly[1], poly[3], poly[2]]

    def bounding_box(self, cam=0):
        """ Returns the bounding box for the camera *cam* as a :class:`Rect`
            object.
        """
        return Rect(*self.bounding_boxes[cam])
            
        
        
class StateSet(Data):
    """ Subclass of :class:`Data` representing a set of states. It is used to
        represent the output of an object detector prior to any tracking have
        been preformed. It has the  following attributes:
        
            **states**
                A list of the represented states
            **synced_file**
                The filename of a file storing a :class:`SyncedVideos` object with
                the videos from which the tracks originates.
            
    """
    
    def _setup(self):
        self._synced = None
        self._by_timestamp = defaultdict(list)
        if not hasattr(self, 'states'):
            self.states = []
        for d in self.states:
            det = State(**d)
            det._parent = self
            det._calc_missing_data(self.synced)
            self._by_timestamp[det.master_timestamp].append(det)
            
    def _get_data(self):
        data = Data._get_data(self)
        data['states'] = [d._get_data() for d in self]
        return data

    @property
    def synced(self):
        """ A :class:`SyncedVideos` object with the videos from which the tracks
            originates loaded from *synced_file*.
        """
        if self._synced is None:
            self._synced = self._load_child(self.synced_file)        
        return self._synced
        
    def __iter__(self):
        """ Iterate over all states in the set. """
        return iter(self.states)
        
    def __len__(self):
        """ The number of :class:`State` objects in the set """
        return len(self.states)
        
    def __getitem__(self, frame):
        """ Return the :class:`State` objects with frame index *frame* from
            all the :class:`State` objects within the set.
        """
        return self.detections_at_time(self.synced.master_timestamps[frame])
        
    def detections_at_time(self, ts):
        """ Return the :class:`State` objects with frame timestamp *ts* from
            all the :class:`State` objects within the set.
        """
        return self._by_timestamp[ts]
        
        
    def append(self, det):
        """ Append the :class:`State` object *det* to the
            set of states.
        """
        assert det._parent is None
        det._parent = self
        self.states.append(det)
        det._calc_missing_data(self.synced)
        self._by_timestamp[det.master_timestamp].append(det)
        
    @property
    def first_frame(self):
        """ The index of the first frame among all the states in the set. """
        return self.synced.timestamp_index(min(self._by_timestamp.keys()))

    @property
    def last_frame(self):
        """ The index of the last frame among all the states in the set. """
        return self.synced.timestamp_index(max(self._by_timestamp.keys()))   
        
    @property
    def frame_range(self):
        """ Iterator over all the frames between *first_frame* and *last_frame* 
            (inclusive). """
        return range(self.first_frame, self.last_frame + 1)
        
class Events(Data):
    """ Subclass of :class:`Data` representing a set of :class:`Event` objects 
        describing interesting events. It contains the attributes:
        
            **event_types**
                A dictionary containing human readable definitions of the 
                different event type strings used in :class:`Event`.type.
            **events**
                A list of :class:`Event` objects.
    """
    def _setup(self):
        if hasattr(self, 'events'):
            fn = os.path.join(os.path.dirname(self._filename), '_fake_event_file.json')
            self.events = [Event(_filename=fn, **e) for e in self.events]
        else:
            self.events = []
            
    def _get_data(self):
        data = Data._get_data(self)
        data['events'] = [e._get_data() for e in self]
        return data

    def __iter__(self):
        """ Iterate over all the events. """
        return iter(self.events)
        
    def __getitem__(self, idx):
        """ Returns the event with index *idx*. """
        return self.events[idx]
        
    def append(self, e):
        """ Appends the event *e* to the list of events. """
        assert isinstance(e, Event)
        if not hasattr(e, '_filename') and hasattr(self, '_filename'):
            e._filename = os.path.join(os.path.dirname(self._filename), '_fake_event_file.json')
        self.events.append(e)

class Event(Data):
    """ Subclass of :class:`Data` representing an interesting event. It contains
        the attributes:
        
            **preview_file**
                The filename of a file storing a .avi video showing the event
                with the road users involved in the event annotated.
            **trackset_file**
                The filename of a file storing a :class:`TrackSet` object containing
                the trajectories of the road users involved in the event.
            **type**
                A string specifying the type of event as described by
                :class:`Events`.event_types.
    """
    def _setup(self):
        self._trackset = None
    
    @property
    def trackset(self):
        """ A :class:`TrackSet` object containing the trajectories of the road users 
            involved in the event loaded from *trackset_file*.
        """
        if self._trackset is None:
            self._trackset = self._load_child(self.trackset_file)
        return self._trackset
    

class Scene(Data):
    """ Subclass of :class:`Data` containing metadata about the scene. It contains
        a general set of key/value strings that are intended to be human readable. 
        It might for example include:
        
            **date**
                The date when the recordings were made.
            **duration**
                The length of the recordings.
            **location**
                Specifies where in this world the scene is located.
            **number_of_cameras**
                The number of camera angles recorded.
    """

class Done(Exception):
    pass

def hungarian(list1, list2, cost):
    if len(list1) == 0 or len(list2) == 0:
        return [(a, None) for a in list1] + [(None, b) for b in list2]
    from munkres import Munkres
    matrix = [[cost(a, b) for a in list1] for b in list2]
    m = Munkres()
    indexes = m.compute(matrix)
    nomatch1 = set(range(len(list1))).difference([a for b, a in  indexes])
    nomatch2 = set(range(len(list2))).difference([b for b, a in  indexes])
    return [(list1[a], list2[b]) for b, a in  indexes] + \
            [(list1[a], None) for a in nomatch1] + \
            [(None, list2[b]) for b in nomatch2]
    
def generate_latex(results):
    types = list(results.keys())
    types.sort()
    s  =  '\\begin{tabular}{l|%s}' % ('|'.join('l' * len(types))) + '\n'
    s += '&' + '&'.join(types) + '\\\\\n'
    s += '\\hline\n'
    s += 'True tracks     & ' + ' & '.join('%5d' % (results[t].true_tracks) for t in types) + '\\\\\n'
    s += 'Detected tracks & ' + ' & '.join('%5d' % (results[t].matched_tracks) for t in types) + '\\\\\n'    
    s += 'Missed tracks   & ' + ' & '.join('%5d' % (results[t].missed_tracks) for t in types) + '\\\\\n'
    s += 'Extra tracks    & ' + ' & '.join('%5d' % (results[t].extra_tracks) for t in types) + '\\\\\n'        
    s += '\\hline\n'
    s += 'True states     & ' + ' & '.join('%5d' % (results[t].true_states) for t in types) + '\\\\\n'
    s += 'Detected states & ' + ' & '.join('%5d' % (results[t].matched_states) for t in types) + '\\\\\n'    
    s += 'Missed states   & ' + ' & '.join('%5d' % (results[t].missed_states) for t in types) + '\\\\\n'
    s += 'Extra states    & ' + ' & '.join('%5d' % (results[t].extra_states) for t in types) + '\\\\\n'        
    s += '\\end{tabular}\n'
    
    return s
        
    
def main(cmd=None, *args):
    if cmd == 'show' and len(args) == 1:
        load(args[0]).show()
    elif cmd == 'compare' and 2 <= len(args) <= 3:
        th = float(args[2]) if len(args) > 2 else 1.0/3.0
        res = load(args[0]).compare(load(args[1]), th)
        print(res)
        res.plot_dists()
    elif cmd == 'compare_latex' and 2 <= len(args) <= 4:
        th = float(args[2]) if len(args) > 2 else 1.0/3.0
        res = load(args[0]).compare(load(args[1]), th, bytype=True)       
        print()
        print('Results:')
        print(generate_latex(res))
        if len(args) > 3:
            baseres = load(args[0]).compare(load(args[3]), th, bytype=True)        
            print()
            print('Baseline:')
            print(generate_latex(baseres))
            Result.plot_multi_dists(res, show=False)
            Result.plot_multi_dists(baseres, '--')
        else:
            Result.plot_multi_dists(res)

    elif cmd == 'bbcompare' and 2 <= len(args) <= 3:
        th = float(args[2]) if len(args) > 2 else 1.0/3.0
        res = load(args[0]).compare(load(args[1]), th, bounding_box_overlap)
        print(res)
        res.plot_dists()
    elif cmd == 'bbcompare_latex' and 2 <= len(args) <= 3:
        th = float(args[2]) if len(args) > 2 else 1.0/3.0
        res = load(args[0]).compare(load(args[1]), th,
                                    bounding_box_overlap, bytype=True)        
        print(generate_latex(res))
        Result.plot_multi_dists(res)
    elif cmd == 'make_avi' and len(args) == 2:
        load(args[0]).make_avi(args[1])
    else:
        print('Usage: python -mpdtv show <file>')
        print('       python -mpdtv compare <ground_truth> <tracking_result> [<threshold>]')    
        print('       python -mpdtv bbcompare <ground_truth> <tracking_result> [<threshold>]')    
        print('       python -mpdtv compare_latex <ground_truth> <tracking_result> [<threshold>] [<baseline>]')    
        print('       python -mpdtv bbcompare_latex <ground_truth> <tracking_result> [<threshold>]')    
        print('       python -mpdtv make_avi <file> <avifile>')
    
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

