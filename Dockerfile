FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN cat /etc/apt/sources.list | grep multiverse | sed 's/\# //g' >> /etc/apt/sources.list

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    libpng12-0 libgtk2.0 \
    git mercurial subversion \
    fonts-freefont-ttf fonts-ubuntu-font-family-console ttf-ubuntu-font-family edubuntu-fonts fonts-ubuntu-title fonts-liberation fonts-arkpandora \
    font-manager cifs-utils vim && \
    apt-get clean

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

RUN conda upgrade anaconda && \
    conda install seaborn bcolz && \
    conda install -c menpo opencv3==3.2.0 && \
    conda install -c conda-forge nb_conda_kernels

RUN pip uninstall -y html5lib && pip install tensorflow-gpu==1.4.1 
    
RUN cd / && git clone https://github.com/fchollet/keras.git && cd keras && git checkout 507374c8 && pip install . && cd .. && rm -r keras

RUN pip install click==6.6 pudb==2018.1

RUN apt-get update && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev graphviz software-properties-common && \
    apt-get clean

#RUN mv /opt/conda/lib/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6_bak && mv /opt/conda/lib/libgomp.so.1 /opt/conda/lib/libgomp.so.1_bak

RUN pip install -I numpy==1.14.3 tqdm==4.26.0 imageio==2.3.0 line_profiler==2.1.2
RUN echo "import imageio\nimageio.plugins.ffmpeg.download()" | python
RUN echo 'alias prof="kernprof -l -v"' >> ~/.bashrc

RUN pip install dask==1.1.0

RUN pip install pydot==1.4.1 connexion==1.5.3  munkres==1.0.12 flask==1.0.2

RUN add-apt-repository ppa:stebbins/handbrake-releases && apt-get update && apt-get install -y handbrake-cli && \
    apt-get clean

ENV TF_CPP_MIN_LOG_LEVEL 2

WORKDIR "/code"

COPY *.py strudl.yaml /code/
COPY webui /code/webui
COPY pdtv /code/pdtv
COPY tests /code/tests
COPY test_data /code/test_data
RUN mkdir /code/data

ENV PYTHONPATH=/code
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "python",  "server.py" ]

