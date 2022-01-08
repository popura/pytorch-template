FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
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
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

# --- Install Pythoon ---
RUN apt update -y && apt upgrade -y
RUN apt install -y \
    tk-dev \
    build-essential \
    libffi-dev libssl-dev \
    zlib1g-dev liblzma-dev \
    libbz2-dev libreadline-dev libsqlite3-dev && \
    wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tar.xz && \
    tar xJf Python-3.9.9.tar.xz && \
    cd Python-3.9.9 && \
    ./configure && \
    make && make install && \
    cd ../ && \
    python3 -m pip install --upgrade pip

# --- Install PyTorch ---
RUN pip3 install \
    torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# --- Root User Settings in Container ---
ARG ROOT_PASSWORD="password"
RUN echo "root:$ROOT_PASSWORD" | chpasswd

