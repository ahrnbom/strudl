## Folder structure and file formats

STRUDL stores lots of files in lots of fascinating formats, in lots of folders. Here is a rundown.

First of all, all data that STRUDL creates are stored in a folder which is called `/data/`. It does not have to be the actual path `/data/` on the host computer, it can be any folder anywhere. The Docker image will mount this so that for the STRUDL program, it looks like it is in the path `/data/`.

Datasets are stored in `/data/datasets/DATASETNAME/`. A dataset folder has the following folders:

1. `videos/` containing videos as .mkv files. These are recoded during the `import_videos` job, to be in a desired resolution and to have a reasonable compression level.
1. `logs/` containing `.log` text files specifying which actual time each frame corresponds to
1. `klt/` containing `.pklz` files in the original point track format (see below), as well as `.mp4` files showing the point tracks drawn over the videos.
1. `objects/` containing two subfolders, `train/` and `test/`, each containing subfolders for each video and then finally training images as `.png` as well as `.txt` annotation files (see below). A file `frames.log` specifies which frames the images come from.

Directly in the datasets folder should have the following files:

1. `classes.txt` which contains the class names, one per line (please avoid any characters other than lowercase letters)
1. `config.txt` which contains a dataset configuration (the format is specified in `strudl.yaml` under DatasetConfig)
1. an image `mask.png` which should be of the same resolution as the videos and be transparent except for the regions which are to be ignored.
1. `calib.tacal` which is a text file containing camera calibration parameters of the TSAI camera model

Each training run has a folder, located as `data/runs/DATASETNAME_RUNNAME/`, containing the following:

1. `checkpoints/` containing weight files for the trained object detector
1. `csv/` containing `.csv` files containing the detected objects, in the resolution of the detector
1. `detections_world/` which contains `.csv` files for the detected objects in both pixel coordinates (in the resolution of the videos) and in world coordinates, as well as `.pklz` files containing point tracks in the per-detection point track format (see below)
1. `tracks/` containing pixel-coordinate tracks in `.pklz` files in the pixel coordinate tracks format (see below) and `.mp4` videos visualizing the tracks
1. `tracks_world/` containing world-coordinate tracks in `.pklz` files in the world coordinate tracks format (see below) and `.mp4` videos visualizing the tracks
1. `track_zips/` containing `.zip` files containing all the tracks in text formats (as made by `track_formats.py`)

### File formats

A general note: `.pklz` files are read and written via the `storage.py` module. They are compressed pickle objects.

#### Original point track format

A `.pklz` file which contains a list of `Track` objects. `Track` objects are lists but with a special `.id_num` attribute. The list contains tuples like `(FRAMENUMBER, X, Y)` where `X` and `Y` are in the point track resolution.

#### Annotation file format

The text files can be empty, which means no objects for this image. If there are objects, each line in the text file corresponds to one object. The format is as follows:

`CLASSID CENTERX CENTERY WIDTH HEIGHT px:X1,X2,X3,X4 py:Y1,Y2,Y3,Y4 CLASSNAME`

The `px:` and `py:` could be missing and training will still work. All coordinates here are between 0 and 1. `CLASSID` starts at 1 and is the number of the class in the `classes.txt` file.

#### Per-detection point track format

A `.pklz` file containing a `dict` which as keys uses detection IDs. For each detection, it then stores a list of point tracks that go through this detected object. The point tracks are `dict`s using frame numbers as keys, with values as tuples `(X,Y)` in the video resolution, in pixel coordinates.

#### Pixel coordinates track format

A `.pklz` file containing a list of `DetTrack` objects (from the `tracking.py` module). They store their tracks in the `.history` attribute, which is a list of tuples of format `(FRAMENUMBER, X, Y, WIDTH, HEIGHT, BY_UPDATE)` where the coordinates are in the video resolution and `BY_UPDATE` is `True` if a detection was used to set this position, and `False` if the position was made by following point tracks.

#### World coordinates track format

A `.pklz` file containing a list of `WorldTrack` objects (from the `tracking_world.py` module). They store their tracks in the `.history` attribute, which is a list of tuples of format `(FRAMENUMBER, TIME, X, Y, DELTAX, DELTAY, SPEED, BY_UPDATE)` where the coordinates are in world coordinates (metres, and metres/second) and `BY_UPDATE` is `False` if this frame's psoition was extrapolated and `True` if an detection caused the track's position.

#### Export track formats

When tracks are exported, they are stored in text formats. Currently there is one format available called "custom_text" which has the following format:

```
Track ID: _TRACKID_, class: _CLASSNAME_
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_ 
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_
  
Track ID: _TRACKID_, class: _CLASSNAME_
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_ 
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_
  fn: _FRAMENUMBER_, t: _TIME_, x: _X_, y: _Y_, dx: _DX_, dy: _DY_, sp: _SPEED_
```

There is also a csv format which is similar, but csv. This means that track ID and class is repeated for every line in a track.

If some other format is desired, modify `tracks_format.py` (add a new function similar to `convert_track_custom_text` and make sure it is used in `format_tracks_from_file`, while also changing the `all_track_formats` list) and `strudl.yaml` (change the `tracksFormat` parameter to have your new format in the enum).
