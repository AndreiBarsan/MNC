FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer andrei.ioan.barsan@gmail.com

# Build and run this with 'nvidia-docker'. If you forget to do so, the build
# will NOT fail, but you will start getting strange issues when attempting
# to run Caffe or any of the tests.

# Based on the official Caffe Dockerfile, tweaked to support the Multi-Task
# Network Cascades software.
#  * Includes Python 2.7 bindings
#  * Includes tons of small fixes for cublas, hdf5, etc.

# Protip: Do NOT attempt to run any CUDA code (including tests) as part of the
# docker build process. It will NOT work. You need to build everything, and
# then run the tests separately using something like:
#
#   nvidia-docker run -ti <built-image-name> bash -c 'cd /opt/caffe/MNC/caffe-mnc && make runtest'

# Use this if you want your code to only use a specific GPU.
#   ENV CUDA_VISIBLE_DEVICES=1

# TODO(andreib): Try cuDNN 7 and see if inference is faster.
# TODO(andreib): Try CUDA 9 and see if inference faster.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        vim \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# TODO-LOW(andreib): Do we really absolutely need OpenCV?
RUN pip install --upgrade pip && \
    pip install Cython numpy scipy scikit-image matplotlib h5py nose pandas \
    protobuf python-gflags Pillow six python-dateutil pyyaml \
    easydict opencv-python

ENV MNC_ROOT=/opt/mnc
ENV CAFFE_ROOT=/opt/mnc/caffe-mnc
WORKDIR $MNC_ROOT

# Support multi-GPU systems
RUN git clone https://github.com/NVIDIA/nccl.git
RUN cd nccl && make -j$(nproc) install && cd .. && rm -rf nccl

# Access our project-specific configs, such as the Makefile.config required for
# building Caffe.
ADD files /usr/local/files

# Grab the native parts of the code which rarely need rebuilding
# TODO(andreib): Do we really need the copy?
ADD caffe-mnc /usr/local/MNC/caffe-mnc
ADD lib       /usr/local/MNC/lib
ADD data      /usr/local/MNC/data
ADD models    /usr/local/MNC/models
RUN cp -R /usr/local/MNC/caffe-mnc .
RUN cp -R /usr/local/MNC/lib .
RUN cp -R /usr/local/MNC/data .
RUN cp -R /usr/local/MNC/models .

RUN echo "Quietly building native components for Python helpers..."
RUN cd lib && make -j4 >/dev/null 2>&1

# The 'rm' is required since otherwise the remaining tests don't pass.
RUN cd caffe-mnc && \
    cp /usr/local/files/Makefile.config . && \
    rm -f src/caffe/test/test_smooth_L1_loss_layer.cpp

# More duct tape, grumble, grumble...
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so /usr/lib/x86_64-linux-gnu/libhdf5.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

RUN echo "Building Caffe using $(nproc) CPUs..."
RUN cd caffe-mnc && make all -j$(nproc)
RUN cd caffe-mnc && make pycaffe -j$(nproc)
RUN cd caffe-mnc && make test -j$(nproc)

# Do NOT run the tests here (make test builds the tests; make runtest would run
# them). Running CUDA code when building an image does NOT work.
# See the top of this file for info on how to run tests.

RUN echo "Fetching pre-trained models. This may take a few minutes..."
RUN data/scripts/fetch_mnc_model.sh >/dev/null 2>&1

# Only mount the tools now so that tweaks to the tool scripts do not trigger
# full Caffe rebuilds. (This could also be achieved by caching Caffe build
# artifacts on a mounted host folder.)
ADD tools /usr/local/MNC/tools
RUN cp -R /usr/local/MNC/tools .

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace