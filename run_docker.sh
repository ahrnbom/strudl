#!/usr/bin/env bash

# Run this script with a parameter to provide a custom path to your data folder.
DATAPATH=$PWD/data/
if [ $# == 1 ] 
then
    DATAPATH=$1
fi

echo "Using data from $DATAPATH"

docker build -t strudl .
docker rm -f strudl_sess
nvidia-docker run --name strudl_sess -p 80:80 -v $PWD:/code/ -v $DATAPATH:/data/ --mount type=bind,src=/media,dst=/usb,bind-propagation=rslave -it strudl bash
# Mounting /media as an rslave allows STRUDL to access USB drives inserted after the docker image starts. A simple -v does not support this, for some reason.
