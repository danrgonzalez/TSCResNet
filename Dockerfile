FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/

ADD environment.yml /environment.yml

RUN conda update -n base conda
RUN conda env create -f environment.yml && conda clean -a

#ADD ./app /app
#ADD ./notebooks /notebooks
#ADD ./config /config
#ADD ./models /models

WORKDIR /

ENTRYPOINT source activate TSCResNet && jupyter notebook

#ENTRYPOINT source activate TSCResNet && python app/main.py
