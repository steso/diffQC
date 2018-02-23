FROM bids/base_fsl

# Install python and nibabel
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install nibabel && \
    apt-get remove -y python3-pip && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENV PYTHONPATH=""

RUN apt-get update && apt-get install -y curl git perl-modules python software-properties-common tar unzip wget
RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y nodejs

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y g++-5

RUN apt-get install -y libeigen3-dev zlib1g-dev

ENV CXX=/usr/bin/g++-5

RUN git clone https://github.com/MRtrix3/mrtrix3.git mrtrix3 && \
    cd mrtrix3 && git checkout 54faeb61 && \
    python configure -nogui && \
    NUMBER_OF_PROCESSORS=1 python build && \
    git describe --tags > /mrtrix3_version

COPY run.py /run.py

COPY version /version

ENTRYPOINT ["/run.py"]
