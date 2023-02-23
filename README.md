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

# Using poetry
  1. Change variables in `pyproject.toml` as needed

    - e.g., name, version, description, and authors
  1. Remove unrequired dependencies in `pyproject.toml`
  1. Create a vertual environment and install dependencies by

     ```
      poetry install
     ```
  1. Add required dependencies by

     ```
      poetry add <library name>
     ```

# Developing your package
  1. For prototyping, make `notebooks/local` directory and add jupyter notebook into the directory.
  1. For developing, add source files into `src` directory
  1. For testing, add test codes into `test` directory
  1. For evaluation, add script files into `scripts` directory

