FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
MAINTAINER Dinesh Daultani <dinesh.daultani@gmail.com>

# Temp fix for current nvidia key issue
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list \
	&& rm /etc/apt/sources.list.d/nvidia-ml.list \
	&& dpkg -i cuda-keyring_1.0-1_all.deb

ARG DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
     apt-get install -y wget vim git zip && \ 
     apt-get install -y sudo software-properties-common systemd-sysv 

# Installing Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# Install libraries
RUN conda create -n py38 python=3.8
SHELL ["/bin/bash", "-c"]
RUN source activate py38 && \
		echo $(python -V) && \
		conda install -y scikit-learn && \
		conda install -y pytorch==1.12.0 torchvision==0.13.0 -c pytorch

COPY requirements.txt /tmp/
RUN source activate py38 && \
		pip install -r /tmp/requirements.txt