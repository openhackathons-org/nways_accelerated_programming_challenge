# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $  sudo docker build -f nways_Dockerfile -t nways:cf .
# To run: $ sudo docker run --rm -it --runtime nvidia -p 8888:8888 nways:cf
# Finally, open http://localhost:8888/

FROM nvcr.io/nvidia/nvhpc:24.1-devel-cuda_multi-ubuntu22.04

RUN apt-get -y update && \
        DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential libtbb-dev && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --upgrade pip &&\
        pip3 install numpy &&\
        pip3 install jupyterlab &&\
        pip3 install ipywidgets &&\
        pip3 install gdown

############################################

# TO COPY the data
COPY English/ /labs/

#################################################
ENV PATH="/usr/local/bin:/opt/anaconda3/bin:/usr/bin:$PATH"
#################################################

WORKDIR /labs
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/labs
