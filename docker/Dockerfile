FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# --- Install Packages ---
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    sudo make zip unzip git curl wget vim && \
    apt autoremove -y && \
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
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tar.xz && \
    tar xJf Python-3.10.4.tar.xz && \
    cd Python-3.10.4 && \
    ./configure && \
    make && make install && \
    cd ../ && \
    python3 -m pip install --upgrade pip

# --- Install PyTorch ---
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url \
     https://download.pytorch.org/whl/cu113

# --- Root User Settings in Container ---
ARG ROOT_PASSWORD="password"
RUN echo "root:$ROOT_PASSWORD" | chpasswd

# --- User Settings in Container ---
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get -y install gosu && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

