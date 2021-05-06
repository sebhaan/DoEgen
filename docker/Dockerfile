# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

# Install conda environment
RUN conda create -n doegen_app python=3.7

# Activate conda environment and install swig and numpy
ENV PATH /opt/conda/envs/doegen_app/bin:$PATH
RUN /bin/bash -c ". activate doegen_app" && \
    conda install --yes swig

# Install DoEgen (incl OApackage) from PyPi
# Numpy and gcc must be explicitely installed before installing OApackage
RUN apt-get update && apt-get install -y g++
RUN pip install numpy 
# DoEgen install might take a few minutes since OApackage wheel building very long
RUN pip install DoEgen

WORKDIR project/
ENTRYPOINT ["python", "-m"]

# Give default arguments, in case none are supplied on
# the command-line, e.g.
CMD ["doegen.init_tests"]


#HOW TO BUILD (IN SHELL), e.g.:
#docker build -t doegen-app:v1 .

#HOW TO RUN (IN SHELL), e.g.:
#docker run -it -v /Users/seb/CTDS/Projects/EDO/code/DoEgen/docker/project:/project doegen-app:v1 doegen.doegen ...yaml
#docker run -it -v /Users/seb/CTDS/Projects/EDO/code/DoEgen/docker/project:/project doegen-app:v1 doegen.doeval ...yaml
