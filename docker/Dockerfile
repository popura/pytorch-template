FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER Yuma.K

# --- Install Packages with Apt-Get ---
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
RUN apt install -y python python-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install --upgrade pip
RUN rm get-pip.py

# --- Install Python Requirements ---
RUN pip install setproctitle==1.1.10 numpy==1.12.0 \
    scipy==0.18.1 pyglib==0.1 Cython==0.17 opencv-python \
    Pillow
RUN pip install scikit_image==0.14 \
    tensorflow_gpu==1.1.0 python_gflags==3.1.1 \
    python_magic==0.4.13

# --- Root User Settings in Container ---
ARG ROOT_PASSWORD="password"
RUN echo "root:$ROOT_PASSWORD" | chpasswd

# --- User Settings in Container ---
# NOTE: It is assumed that root user has UID=10000:GID=100 in host computer.
ARG GROUP_ID=413
ARG GROUP_NAME="Domain"
RUN groupadd -g $GROUP_ID $GROUP_NAME
# Main User Settings
ARG MAIN_USER_ID=308
ARG MAIN_USER_NAME="sayaka"
RUN useradd --create-home --shell /bin/bash -g $GROUP_NAME -u $MAIN_USER_ID $MAIN_USER_NAME && \
    echo "$MAIN_USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

