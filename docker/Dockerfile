FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="Yuma Kinoshita"

ARG PROJECT_NAME=pytorch-template
ARG PYTHON_VERSION=3.9.11
ARG APPLICATION_DIRECTORY=/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# --- Change repository ---
# RUN sed -i.org -e 's|archive.ubuntu.com|ubuntutym.u-toyama.ac.jp|g' /etc/apt/sources.list && \
#     apt-get clean

# --- Install Packages with Apt ---
RUN apt-get update -y && apt-get upgrade -y --fix-missing
RUN apt-get install -y --fix-missing \
    sudo ssh \
    build-essential \
    zsh tmux make zip unzip git curl wget vim tree htop \
    graphviz && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# --- Install Python ---
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    tk-dev \
    build-essential \
    libffi-dev libssl-dev \
    zlib1g-dev liblzma-dev \
    libbz2-dev libreadline-dev libsqlite3-dev && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xJf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make && make install && \
    cd ../ && \
    python3 -m pip install --upgrade pip setuptools requests && \
    python3 -m pip install poetry && \
    rm -rf Python-${PYTHON_VERSION}.tar.xz && \
    rm -rf Python-${PYTHON_VERSION}

WORKDIR ${APPLICATION_DIRECTORY}
COPY pyproject.toml poetry.toml poetry.lock* .
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "ture" && poetry install || echo "skip to run poetry install."
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."
COPY docker/jupyter/jupyter_notebook_config.py /${HOME}/.jupyter/jupyter_notebook_config.py

