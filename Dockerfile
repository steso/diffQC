FROM bids/base_fsl

# Install python and nibabel
RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN apt-get install -y git

RUN sudo apt-get install build-essential checkinstall -y
RUN	sudo apt-get install libreadline-gplv2-dev -y
RUN sudo apt-get install libncursesw5-dev
RUN sudo apt-get install libssl-dev -y
RUN sudo apt-get install libsqlite3-dev -y
RUN sudo apt-get install tk-dev -y
RUN sudo apt-get install libgdbm-dev -y
RUN sudo apt-get install libc6-dev -y
RUN sudo apt-get install libbz2-dev -y

RUN cd /usr/src && \
	wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz && \
	sudo tar xzf Python-3.5.2.tgz && \
	cd Python-3.5.2 && \
	sudo ./configure && \
	sudo make altinstall

RUN git clone https://github.com/cython/cython.git cython && \
    cd cython && \
    git checkout 8ad16fc && \
    pip3.5 install . && \
    cd ..

RUN git clone https://github.com/numpy/numpy.git numpy && \
	cd numpy && \
	git checkout 6914bb4 && \
	pip3.5 install . && \
	cd ..

RUN git clone https://github.com/nipy/nibabel.git nibabel && \
	cd nibabel && \
	git checkout 6fb7538 && \
	pip3.5 install . && \
	cd ..

#RUN apt-get remove -y python3-pip && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

## install more python libraries

## RUN apt-get update && \
##     apt-get install -y python3 python3-pip python3-scipy libfreetype6-dev libatlas-dev libatlas3gf-base libhdf5-dev

## RUN update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3 && \
##     update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3

## RUN pip3 install numpy scikit-learn matplotlib dipy && \
##     apt-get remove -y python3-pip && \
##     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#RUN apt-get install -y python3-pip libfreetype6-dev libpng12-dev python3-tk
#RUN pip3 install --upgrade pip

#RUN sudo apt-get update
#RUN sudo apt-get upgrade -y
#RUN wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
#RUN tar -xvzf Python-3.5.6.tgz
#RUN cd Python-3.5.6 && \
#	./configure --enable-loadable-sqlite-extensions --enable-optimizations && \
#	make && \
#	#make test && \
#	sudo make install && \
#	sudo ldconfig

RUN git clone https://github.com/matplotlib/matplotlib.git matplotlib && \
	cd matplotlib && \
	git checkout 682e519 && \
	pip3.5	install . && \
	cd ..

RUN apt-get install -y python3-scipy libfreetype6-dev libatlas-dev libatlas3gf-base libhdf5-dev

RUN sudo apt-get install gfortran libopenblas-dev liblapack-dev -y
RUN sudo pip3.5 install scipy==1.2.1

RUN git clone https://github.com/scikit-learn/scikit-learn.git scikit-learn && \
	cd scikit-learn && \
	git checkout 8c439fb && \
	pip3.5 install . && \
	cd ..

RUN pip3.5 install --upgrade pip==20.1.0
RUN git clone https://github.com/dipy/dipy.git dipy && \
	cd dipy && \
	git checkout 30a57a6 && \
	pip3.5 install . && \
	cd ..
#RUN pip3.5 install scikit-image

RUN git clone https://github.com/pandas-dev/pandas.git pandas && \
	cd pandas && \
	git checkout 3147a86 && \
	pip3.5 install . && \
	cd ..

RUN git clone https://github.com/scikit-image/scikit-image.git scikit && \
	cd scikit && \
	git checkout d0edde6 && \
	pip3.5 install . && \
	cd ..

#RUN pip3.5 install pandas --no-build-isolation
#RUN pip3.5 install statsmodels

RUN git clone https://github.com/statsmodels/statsmodels.git statsmodel && \
	cd statsmodel && \
	git checkout 40e4f56 && \
	pip3.5 install . && \
	cd ..

COPY run.py /run.py
COPY diffqc /diffqc/

COPY version /version

ENTRYPOINT ["/run.py"]
