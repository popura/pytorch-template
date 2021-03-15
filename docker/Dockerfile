FROM nvidia/cuda:10.2-devel-ubuntu18.04
MAINTAINER Yuma.K

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
RUN apt install -y python python-dev

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

