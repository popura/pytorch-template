# Development Environment Construction
#
# 2021/03/15 Yuma Kinoshita
# 2019/8/4 Yuma Kinoshita
# 2017/3/6 Yousuke Sugawara

USER_NAME = 'yuma.k'
HOST_PATH = '/home/yuma.k/git/DeepUPE'
CONTAINER_PATH = '/home/yuma.k/git/DeepUPE'
ROOT_PASSWORD = 'password'
MAIN_USER_ID = 52801133
MAIN_USER_NAME = 'sayaka'
MAIN_USER_PASSWORD = 'password'

all: build run

build:
	docker build -t nvidia/cuda:sample ./ \
	--build-arg ROOT_PASSWORD=$(ROOT_PASSWORD) \
	--build-arg MAIN_USER_ID=$(MAIN_USER_ID) \
	--build-arg MAIN_USER_NAME=$(MAIN_USER_NAME)

run:
	docker run -it --rm --gpus all -u $(USER_NAME) -v $(HOST_PATH):$(CONTAINER_PATH) nvidia/cuda:sample /bin/bash
