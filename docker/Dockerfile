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
RUN apt-get update && apt-get -y install gosu
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
