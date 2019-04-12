# STRUDL: Surveillance Tracking Using Deep Learning

STRUDL is an open-source and free framework for tracking objects in videos filmed by **static surveillance cameras**. It uses a deep learining object detector, camera calibration and tracking to create trajectories of e.g. road users, in world coordinates. It was designed to faciliate traffic safety analysis, using modern computer vision and deep learning, rather than the traditional methods commonly used despite their many flaws. By creating trajectories in world coordinates, truly meaningful metrics and safety measures can be computed. The paper behind this research will hopefully be released soon! It is currently under review.

STRUDL was developed as a part of the [InDeV project](https://www.indev-project.eu). 

[![InDeV Logo](https://www.indev-project.eu/SiteGlobals/StyleBundles/CSS/screen/InDeV/indevSub_logo.jpg?__blob=normal&v=10)](https://www.indev-project.eu)

It provides a Web UI that attempts to make it easy to use, even without too much knowledge in computer vision and deep learning. 

It uses [Docker](https://www.docker.com/) for sandboxing and handling dependencies, to make installation easier. The code is designed to be modular, so that new features can be added fairly easily without destroying everything else. It uses [Swagger](https://swagger.io/) and [Connexion](https://github.com/zalando/connexion) to provide a REST API and parts of the Web UI. Because all the functionality can be used via a REST API, users can create their own interfaces if they prefer, to simplify some actions and automate tasks.

Some more information about how STRUDL works internally can be found [here](details.md).

Got any issues with this software? Feel free to [open an issue, if there isn't one already for your problem](https://github.com/ahrnbom/strudl/issues)!

### Workflow
1. Import videos alongside log files specifying the times for each frame in the videos
2. Annotate images
3. Train an object detector
4. Provide camera calibration
5. Perform tracking in world coordinates
6. Download the tracks, and analyze them with whatever tools you like

### Some features
1. Take advantage of modern deep learning without extensive knowledge of the technology
1. Easily make object detection annotations via custom web interface
1. Semi-automatic annotation process, where a detector can be trained on the currently available training data, and those detections can then be used as a starting point for annotations, speeding up the process once a critical number of images has been annotated.
1. When importing videos, typically from a USB drive, videos can be recoded to arbitrarily long/short clips, and different resolutions.
1. Tracking in world coordinates, which can optimized if the user provides ground truth tracks made by [T-Analyst](https://bitbucket.org/TrafficAndRoads/tanalyst/wiki/Manual). In our experience, this is not necessary and the default parameters work OK for most scenarios.
1. Full trajectories can be obtained as text files (e.g. csv), which the user can then filter, sort and present, using any tools they prefer. This allows e.g. complex traffic analysis.
1. Visualize each step in the process in summary video files, to better understand what goes wrong when something goes wrong.

### Data assumptions
1. Data should be in video files, filmed from static (non-moving) surveillance cameras.
1. A TSAI-calibration should exist, allowing translation between pixel coordinates and world coordinates
1. For each video file, there should be a log file specifying the time for each frame. This allows more accurate measures of time which is useful in tracking. If your videos do not have this information, simply generate such text files using some approximate method of your choice. The log files should be text files in the `.log` format, where each line looks like this: `00158 2017 05 16 01:28:52.642`, where the first number (with five characters) is the frame number, followed by the year, month, day, hour, minute, second and milliseconds.

### Requirements

1. A Linux computer with a powerful, modern NVidia GPU
1. [NVidia CUDA](https://developer.nvidia.com/cuda-downloads), STRUDL is made for CUDA 8.0 but the container runs fine on a CUAD 10.0 host
1. [docker](https://docs.docker.com/install/)
1. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### Installation
The following terminal commands should work on Ubuntu, assuming the requirements are installed correctly. The first thing to do is to run the testsuit to ensure your requirements is set up properly. This will also download strudl if needed:

```
nvidia-docker run -ti ahrnbom/strudl py.test -v
```

A `data` folder has to be created. It can be put anywhere you want, like on a different disk. This could be useful as the `data` folder can get quite large if large amounts of video are to be processed.

```
mkdir -p /path/of/your/choosing/strudl/data
```

Download the starter script from [here](https://raw.githubusercontent.com/ahrnbom/strudl/master/start_strudl.sh) and place it next to the data directory and run it to start strudl
```
/path/of/your/choosing/strudl/start_strudl.sh
```

If you want to upgrade to the latest version before starting add the -u option:
```
/path/of/your/choosing/strudl/start_strudl.sh -u
```

Visit the host computer via a web browser to see the Web UI and interact with it. For example, if you're using the web browser on the same computer, visit `localhost` in a web browser like Firefox.

### Security notice
This software has not been designed with maximum security in mind. It is recommended to run it in a local network behind a firewall. While docker does provide some sandboxing, this code is not "battle tested" and it should not be assumed to be safe. Leaving the port open to the internet could compromise your computer. One possible security flaw is that your computer's `/media` folder is being made available in the docker container, to simplify importing videos via e.g. USB. This can be changed by modifying `run_docker.sh`.

### Future work
There's always more to do! On our to-do list contains, among other things, the following:

1. Simplification of the Web UI, to make it easier to understand what has been done and what to do next
1. SSD code is currently based on [this port by Rykov8](https://github.com/rykov8/ssd_keras). It might be a good idea to change to [this one instead, by Pierluigiferrari](https://github.com/pierluigiferrari/ssd_keras), which is more nicely documented and runs NMS on the GPU.
1. Code should become more readable and better commented (this started as, and in many ways still is, experimental research code)
1. Different tracking algorithms should be examined, possibly replacing the simplistic one currently used
1. Reducing the size of the docker container (currently around 20 GB)
1. More easter eggs, jokes and memes 
