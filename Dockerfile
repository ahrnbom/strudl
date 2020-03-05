FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN cat /etc/apt/sources.list | grep multiverse | sed 's/\# //g' >> /etc/apt/sources.list

RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:stebbins/handbrake-releases && apt-get update && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    libpng12-0 libgtk2.0 \
    git \
    python3-dev python3-pip python3-tk \
    fonts-freefont-ttf fonts-ubuntu-font-family-console ttf-ubuntu-font-family \
    edubuntu-fonts fonts-ubuntu-title fonts-liberation fonts-arkpandora \
    font-manager cifs-utils ffmpeg \
    handbrake-cli libavcodec-dev libavformat-dev libswscale-dev graphviz libxtst6 && \
    apt-get clean

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN pip3 install --upgrade pip && pip install tensorflow-gpu==1.4.1 && \
    pip install -I numpy==1.14.3 && \
    pip install click==6.6 pudb==2018.1 tqdm==4.26.0 imageio==2.3.0 \
    line_profiler==2.1.2 dask==1.1.0 pydot==1.4.1 connexion==1.5.3 \
    munkres==1.0.12 flask==1.0.2 opencv-contrib-python==3.2.0.8 pytest==4.6.3 \
    pandas==0.23.4 psutil==5.2.2 scipy==0.19.0 matplotlib==2.0.2 h5py==2.7.0 \
    jsonschema==2.6.0 werkzeug==0.16.0
    
RUN cd / && git clone https://github.com/fchollet/keras.git && cd keras && git checkout 507374c8 && pip install . && cd .. && rm -r keras

RUN echo 'alias prof="kernprof -l -v"' >> /etc/bash.bashrc
RUN echo 'alias python="python3"' >> /etc/bash.bashrc

ENV TF_CPP_MIN_LOG_LEVEL 2

WORKDIR "/code"

COPY *.py strudl.yaml /code/
COPY webui /code/webui
COPY pdtv /code/pdtv
COPY tests /code/tests
COPY test_data /code/test_data
RUN mkdir /data

ENV PYTHONPATH=/code
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "python3",  "server.py" ]

