FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt update
RUN apt install -y screen
RUN apt install -y build-essential
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install libopenmpi-dev
RUN git clone git@github.com:2019ChenGong/DPImageBench.git
RUN cd DPImageBench; pip install -r requirements_cuda12.1.txt
RUN conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
RUN conda install mpi4py
RUN cd opacus; pip install -e .; cd ..
RUN cd models/PE/improved-diffusion; pip install -e .; cd ..; cd ..; cd ..