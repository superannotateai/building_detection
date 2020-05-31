FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER martun

ENV DEBIAN_FRONTEND noninteractive

# Install dependent packages via apt-get
RUN apt-get -y update &&\
    echo ">>>>> packages for building python" &&\
    apt-get --no-install-recommends -y install \
      g++ \
      libsqlite3-dev \
      libssl-dev \
      libreadline-dev \
      libncurses5-dev \
      lzma-dev \
      liblzma-dev \
      libbz2-dev \
      libz-dev \
      libgdbm-dev \
      build-essential \
      cmake \
      make \
      wget \
      unzip \
      &&\
    echo ">>>>> packages for building python packages" &&\
    apt-get --no-install-recommends -y install \
      libblas-dev \
      liblapack-dev \
      libpng-dev \
      libfreetype6-dev \
      pkg-config \
      ca-certificates \
      libhdf5-serial-dev \
      postgresql \
      libpq-dev \
      curl \
      python3-pip \
      python3-setuptools \
      python-dev \
      vim \
      &&\
    apt-get clean

# Remove next line after a while, we need it just because we don't want to re-run the proprocessing.
RUN apt-get --no-install-recommends -y install libblosc-dev

##### Install MiniConda ###########################
RUN wget -P /root https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x /root/Miniconda3-latest-Linux-x86_64.sh
RUN /root/Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH "/root/miniconda3/bin:$PATH"

###### Install other packages ###########################
RUN conda init bash
RUN conda config --add channels conda-forge
RUN conda config --add channels pytorch
RUN conda update --all
RUN conda install hdf5=1.10.5 opencv click pandas scikit-image \
	rasterio shapely cudnn h5py fiona 
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install sklearn


# Looks like I can't install tensorboard without tensorflow. It must be possible, but does not work.
# RUN pip3 install tensorflow-gpu==1.14
RUN pip install tensorboard

# ENV command does not concatenate.
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

###################################################-------------

# -=-=-=- Java -=-=-=-
# (martun): changed oracle-java8-installer to openjdk-8-jdk due to license, oracle doesn't provide the package any more.
RUN apt-get --no-install-recommends -y install software-properties-common &&\
    add-apt-repository ppa:webupd8team/java -y &&\
    apt-get update &&\
    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections &&\
    apt-get install -y openjdk-8-jdk &&\
    apt-get clean

# Deploy OSMdata
# (martun): Removed Khartoum city from the list, can't find its Open Street Map data.
RUN mkdir /root/osmdata
COPY osmdata /root/osmdata/
RUN unzip /root/osmdata/las-vegas_nevada.imposm-shapefiles.zip \
        -d /root/osmdata/las-vegas_nevada_osm/ &&\
    unzip /root/osmdata/shanghai_china.imposm-shapefiles.zip \
        -d /root/osmdata/shanghai_china_osm/ &&\
    unzip /root/osmdata/paris_france.imposm-shapefiles.zip \
        -d /root/osmdata/paris_france_osm/
#    unzip /root/osmdata/ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ.imposm-shapefiles.zip \
#        -d /root/osmdata/ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ_osm/

# Copy and unzip visualizer
COPY code/visualizer-2.0.zip /root/
RUN unzip -d /root/visualizer-2.0 /root/visualizer-2.0.zip

# Deploy codes
COPY code /root/
RUN chmod a+x /root/train.sh &&\
    chmod a+x /root/test.sh

ENV PATH $PATH:/root/

# Install Blosc compression for hdf5. Switched to gzip, so this will be unused after a while.
RUN mkdir -p /root/miniconda3/lib/hdf5/plugin
COPY libH5Zblosc.so /root/miniconda3/lib/hdf5/plugin
COPY libblosc_filter.so /root/miniconda3/lib/hdf5/plugin

# Env
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
WORKDIR /root/
