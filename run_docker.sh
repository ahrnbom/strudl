DATAPATH=$PWD/../data/
if [ $# == 1 ] 
then
    DATAPATH=$1
fi

echo "Using data from $DATAPATH"

docker build -t strudl .
docker rm -f strudl_sess
nvidia-docker run --name strudl_sess -p 80:80 -v $PWD:/code/ -v $DATAPATH:/data/ -v /media/:/usb/ -it strudl
