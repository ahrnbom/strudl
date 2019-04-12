#!/usr/bin/env bash

usage () {
  echo "Usage: $0 [-u] [-d <path>] [-v <version>]"
  echo "       -d   - Specify path to the data directory"
  echo "       -u   - Download (pull) latest version before staring"
  echo "       -v   - Start a specific version of strudl"
  exit
}


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATAPATH=$DIR/data/
TAG=latest

while getopts ":d:uv:" opt; do
  case ${opt} in
    d )
      DATAPATH=$OPTARG
      ;;
    u )
      docker pull ahrnbom/strudl
      ;;
    v )
      TAG=$OPTARG
      ;;
    \? )
      usage
      ;;
  esac
done

shift $((OPTIND -1))
if [ $# != 0 ]; then
    usage
fi



if [ $# == 1 ]; then
    DATAPATH=$1
fi

echo "Using data from $DATAPATH"


nvidia-docker run -p 80:80 -v $DATAPATH:/code/data/ --mount type=bind,src=/media,dst=/usb,bind-propagation=rslave --rm -ti ahrnbom/strudl:$TAG
