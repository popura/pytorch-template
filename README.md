# PyTorch template
A template for PyTorch projects

# Dependencies
- (For docker) rootless docker

# Using docker
  1. Move to `docker` directory

      ```
        cd docker
      ``` 
  1. Change variables in `Makefile` as needed

     - Set IMAGE_TAG and CONTAINER_NAME
  1. Add scripts into `Dockerfile` as needed
  1. Perform the following command for building your docker image

      ```
        make build
      ```
  1. Run a docker container via the following command 

      ```
        make run
      ```
  Note that all changes in the docker container will be deleted when you exit docker container. 
