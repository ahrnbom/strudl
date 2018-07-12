# STRUDL: Surveillance Tracking Using Deep Learning

STRUDL is an open-source and free framework for tracking objects visible from static surveillance cameras. It uses a deep learining object detector, camera calibration and tracking to create trajectories of e.g. road users, in world coordinates. It provides a Web UI that attempts to make it easy to use, even without too much knowledge in computer vision and deep learning. It uses [Docker](https://www.docker.com/) for sandboxing and handling dependencies, to make installation easier.

The code is designed to be modular, so that new features can be added fairly easily without destroying everything else. 

### Workflow
1. Import videos
2. Annotate images
3. Train an object detector
4. Provide camera calibration
5. Perform tracking in world coordinates
6. Download the tracks, and analyze them with whatever tools you like

### Requirements

1. A Linux computer with a powerful, modern NVidia GPU
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### Installation
Note: This has not been quite tested yet!

1. Clone this repo
2. Create a folder called `data` right next to the `strudl` folder
3. `cd` into the `strudl` folder, and run `sudo ./run_docker.sh`
4. If everything works as expected, you should now be inside the docker container, in a folder called `/code`. You should be able to access the `data` folder you created at `/data`.
5. To start the web server, run `python server.py`
6. Visit port 8080 of the host computer via a web browser to see the Web UI and interact with it

### Security notice
This software has not been designed with maximum security in mind. It is recommended to run it in a local network behind a firewall. While docker does provide sandboxing, this code is not "battle tested" and it should not be assumed to be safe. Leaving the port open to the internet could compromise your computer.

### Future work
There's always more to do! On our priority list are, among other things, the following:

1. Simplification of the Web UI, to make it easier to understand what has been done and what to do next
2. It should be possible to re-code the videos into different lengths during the importing step. This could be important for some applications as objects are currently not tracked from one video file to another.
3. It should be possible to train the detector on some annotations, and then use that detector to make semi-automatic annotations that can be inspected and modified by humans. This should speed up the annotation process.
4. It should be possible to import annotated images from one dataset to another during training
5. SSD code is currently based on [this port by Rykov8](https://github.com/rykov8/ssd_keras). It might be a good idea to change to [this one instead, by Pierluigiferrari](https://github.com/pierluigiferrari/ssd_keras), which is more nicely documented and runs NMS on the GPU.
6. Code should become more readable and better commented (this started as, and in many ways still is, experimental research code after all)
 
