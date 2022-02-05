FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/

ADD environment.yml /environment.yml
RUN conda update -n base conda
RUN conda env create -f environment.yml && conda clean -a

#ADD ./app /app
ADD ./notebooks /notebooks
#ADD ./config /config
#ADD ./models /models

WORKDIR /

ENTRYPOINT source activate dissertation && jupyter notebook

#ENTRYPOINT source activate dissertation && python app/main.py
