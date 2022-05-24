# Development Environment Construction
#
# 2021/03/15 Yuma Kinoshita
# 2019/8/4 Yuma Kinoshita
# 2017/3/6 Yousuke Sugawara

USER_NAME = $(USER)
GROUP_NAME = Domain
HOST_PATH = '/home/$(USER_NAME)/<path-to-dir>/pytorch-template'
CONTAINER_PATH = '/home/$(USER_NAME)/pytorch-template'
ROOT_PASSWORD = 'password'
IMAGE_TAG = 'pytorch-template:latest'

all: build run

build:
	docker build -t $(IMAGE_TAG) ./ \
	--build-arg ROOT_PASSWORD=$(ROOT_PASSWORD) \

run:
	docker run -it --rm --gpus all \
	-e USER_NAME=$(USER_NAME) -e GROUP_NAME=$(GROUP_NAME) \
	-e LOCAL_UID=$(shell id -u $(USER)) -e LOCAL_GID=$(shell id -g $(USER)) \
	-v $(HOST_PATH):$(CONTAINER_PATH) \
	$(IMAGE_TAG) /bin/bash
