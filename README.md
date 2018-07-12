# STRUDL: Surveillance Tracking Using Deep Learning

## Note: This framework is still under development, and is not properly tested yet! Please be patient.

STRUDL is an open-source and free framework for tracking objects in videos filmed by static surveillance cameras. It uses a deep learining object detector, camera calibration and tracking to create trajectories of e.g. road users, in world coordinates. It was designed to faciliate traffic safety analysis, using modern computer vision and deep learning, rather than the traditional methods commonly used despite their many flaws. By creating trajectories in world coordinates, truly meaningful metrics and safety measures can be computed. The paper behind this research will hopefully be released soon!

STRUDL was developed as a part of the [InDeV project](https://www.indev-project.eu). 

[![InDeV Logo](https://www.indev-project.eu/SiteGlobals/StyleBundles/CSS/screen/InDeV/indevSub_logo.jpg?__blob=normal&v=10)](https://www.indev-project.eu)

It provides a Web UI that attempts to make it easy to use, even without too much knowledge in computer vision and deep learning. It uses [Docker](https://www.docker.com/) for sandboxing and handling dependencies, to make installation easier.

The code is designed to be modular, so that new features can be added fairly easily without destroying everything else. It uses [Swagger](https://swagger.io/) and [Connexion](https://github.com/zalando/connexion) to provide a REST API and Web UI. 

Some more information about how STRUDL works internally can be found [here](details.md).

Got any issues with this software? Feel free to [open an issue, if there isn't one already for your problem](https://github.com/ahrnbom/strudl/issues)!

### Workflow
1. Import videos
2. Annotate images
3. Train an object detector
4. Provide camera calibration
5. Perform tracking in world coordinates
6. Download the tracks, and analyze them with whatever tools you like

### Some features
1. Easily make object annotations via custom web interface
1. Tracking in world coordinates, which can optimized if the user provides ground truth tracks made by [T-Analyst](http://www.tft.lth.se/en/research/video-analysis/co-operation/software/t-analyst/)
1. 

### Requirements

1. A Linux computer with a powerful, modern NVidia GPU
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
3. git

### Installation
Note: This has not been quite tested yet!

1. In a terminal, navigate to some folder where you want to store STRUDL and its data, e.g. `mkdir ~/strudl && cd ~/strudl/`
1. Clone this repo via `git clone https://github.com/ahrnbom/strudl.git`
2. Create a folder called `data` right next to the `strudl` folder, e.g. `mkdir data`
3. Navigate into the `strudl` folder, and run `run_docker.sh`, which requires sudo-privileges (because docker does), e.g. `cd strudl` and `sudo ./run_docker.sh`
4. If everything works as expected, you should now be inside the docker container, in a folder called `/code`. You should be able to access the `data` folder you created at `/data`.
5. To start the web server, run `python server.py`
6. Visit port 8080 of the host computer via a web browser to see the Web UI and interact with it

### Security notice
This software has not been designed with maximum security in mind. It is recommended to run it in a local network behind a firewall. While docker does provide some sandboxing, this code is not "battle tested" and it should not be assumed to be safe. Leaving the port open to the internet could compromise your computer. The most obvious security flaw is that your computer's `/media` folder is being made available in the docker container, to simplify importing videos via e.g. USB. This can be changed by modifying `run_docker.sh`.

### Future work
There's always more to do! On our to-do list contains, among other things, the following:

1. Simplification of the Web UI, to make it easier to understand what has been done and what to do next
2. It should be possible to re-code the videos into different lengths during the importing step. This could be important for some applications as objects are currently not tracked from one video file to another.
3. It should be possible to train the detector on some annotations, and then use that detector to make semi-automatic annotations that can be inspected and modified by humans. This should speed up the annotation process.
4. It should be possible to import annotated images from one dataset to another during training
5. SSD code is currently based on [this port by Rykov8](https://github.com/rykov8/ssd_keras). It might be a good idea to change to [this one instead, by Pierluigiferrari](https://github.com/pierluigiferrari/ssd_keras), which is more nicely documented and runs NMS on the GPU.
6. Code should become more readable and better commented (this started as, and in many ways still is, experimental research code after all)
7. Different tracking algorithms should be examined, possibly replacing the simplistic one currently used.
8. More easter eggs, jokes and memes. 
