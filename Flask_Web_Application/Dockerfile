FROM nvcr.io/nvidia/pytorch:23.02-py3

ARG OPENCV_VERSION=4.7.0
ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/build

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    wget unzip tzdata \
    build-essential cmake pkg-config \
    libgtk-3-dev libcanberra-gtk3-module \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    gfortran libatlas-base-dev libgtk2.0-dev \
    libgl1-mesa-glx libpq-dev python3-dev python-dev \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-libav \
    gstreamer1.0-alsa \
    gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    python3-gi python3-pip \
    git mc graphviz \
    libtbb2 libtbb-dev libdc1394-22-dev \
    libblas-dev liblapack-dev \
    openssh-client git libprotobuf-dev protobuf-compiler
RUN python3 -m pip install --upgrade pip cmake numpy

WORKDIR ${HOME}
RUN git clone https://github.com/opencv/opencv.git -b $OPENCV_VERSION --depth 1
RUN git clone https://github.com/opencv/opencv_contrib.git -b $OPENCV_VERSION --depth 1
WORKDIR ${HOME}/opencv/build

RUN cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=${HOME}/opencv_contrib/modules \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PROTOBUF=ON \
    -DENABLE_FAST_MATH=ON \
    -DWITH_TBB=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_CUDA=ON \
    -DENABLE_FAST_MATH=1 \
    -DCUDA_FAST_MATH=1 \
    -DWITH_CUBLAS=1 \
    -DCUDA_ARCH_BIN=8.6 \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DPYTHON_EXECUTABLE=$(which python2) \
    -DBUILD_OPENCV_PYTHON2=OFF \
    -DPYTHON3_EXECUTABLE=$(which python3) \
    -DOPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
    -DWITH_FFMPEG=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf ${HOME}/opencv && \
    rm -rf ${HOME}/opencv_contrib && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove

RUN rm -rf /usr/local/lib/python3.8/dist-packages/cv2
RUN mkdir /usr/local/lib/python3.8/dist-packages/cv2
RUN cp -r /usr/lib/python3/dist-packages/cv2/* /usr/local/lib/python3.8/dist-packages/cv2

RUN apt update && apt install -y ffmpeg
RUN pip3 install Flask
WORKDIR /project
RUN git clone https://github.com/vis-opt-group/SCI

WORKDIR /project
ADD models/ /project/models/
ADD utils/ /project/utils/
ADD weights/ /project/weights
ADD data/ /project/data/
ADD layers/ /project/layers/

WORKDIR /project/templates
ADD process_photo.html process_video.html index.html submit_image.html submit_video.html /project/templates/
WORKDIR /project
ADD ./app.py ./process.py /project/

ENTRYPOINT ["python3", "app.py"]