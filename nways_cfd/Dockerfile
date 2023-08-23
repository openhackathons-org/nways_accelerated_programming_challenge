# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $ sudo docker build -t openacc-labs:latest .
# Finally, open http://localhost:8888/

FROM nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu20.04

RUN apt-get -y update && \
        DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential libtbb-dev && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --upgrade pip &&\
        pip3 install jupyterlab &&\
        pip3 install ipywidgets &&\
        pip3 install gdown

############################################

# TO COPY the data
COPY English/ /labs/

#################################################
#ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/11.0/lib64/"
ENV PATH="/usr/local/bin:/opt/anaconda3/bin:/usr/bin:$PATH"
#################################################

WORKDIR /labs
CMD service nginx start && jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs
