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
    cd mrtrix3 && git checkout 57e351e && \
    python configure -nogui && \
    NUMBER_OF_PROCESSORS=1 python build && \
    git describe --tags > /mrtrix3_version

# set mrtrix environment variables
ENV PATH=:/mrtrix3/bin:$PATH
ENV PYTHONPATH=/mrtrix3/lib

# install more python libraries

# RUN apt-get update && \
#     apt-get install -y python3 python3-pip python3-scipy libfreetype6-dev libatlas-dev libatlas3gf-base libhdf5-dev

# RUN update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3 && \
#     update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3

# RUN pip3 install numpy scikit-learn matplotlib dipy && \
#     apt-get remove -y python3-pip && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get install -y python3-pip libfreetype6-dev libpng12-dev python3-tk
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib scipy scikit-learn dipy scikit-image
RUN pip3 install pandas --no-build-isolation

COPY run.py /run.py
COPY diffqc /diffqc/

COPY version /version

ENTRYPOINT ["/run.py"]
