# Development Environment Construction
#
# 2019/8/4 Yuma Kinoshita
# 2017/3/6 Yousuke Sugawara

USER_NAME = 'yuma.k'
HOST_PATH = '/home/yuma.k/git/DeepUPE'
CONTAINER_PATH = '/home/yuma.k/git/DeepUPE'
ROOT_PASSWORD = 'password'
MAIN_USER_ID = 52801133
MAIN_USER_NAME = 'yuma.k'
MAIN_USER_PASSWORD = 'password'

all: build run

build:
	nvidia-docker build -t cuda8/cudnn5:hdrnet ./hdrnet \
	--build-arg ROOT_PASSWORD=$(ROOT_PASSWORD) \
	--build-arg MAIN_USER_ID=$(MAIN_USER_ID) \
	--build-arg MAIN_USER_NAME=$(MAIN_USER_NAME)

run:
	nvidia-docker run -it --rm -u $(USER_NAME) -v $(HOST_PATH):$(CONTAINER_PATH) cuda8/cudnn5:hdrnet /bin/bash
