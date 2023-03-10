FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
MAINTAINER Yuma.K

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# --- Install Packages with Apt ---
RUN apt update -y && apt upgrade -y
RUN apt install -y \
    sudo ssh \
    build-essential \
    zsh tmux make zip unzip git curl wget vim tree htop \
    graphviz && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# --- Install Python ---
RUN apt update -y && apt upgrade -y
RUN apt install -y \
    tk-dev \
    build-essential \
    libffi-dev libssl-dev \
    zlib1g-dev liblzma-dev \
    libbz2-dev libreadline-dev libsqlite3-dev && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*
RUN wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tar.xz && \
    tar xJf Python-3.9.11.tar.xz && \
    cd Python-3.9.11 && \
    ./configure && \
    make && make install && \
    cd ../ && \
    python3 -m pip install --upgrade pip && \
    rm -rf Python-3.9.11.tar.xz && \
    rm -rf Python-3.9.11

# --- Install PyTorch ---
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt
COPY docker/jupyter/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

