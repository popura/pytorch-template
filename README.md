# PyTorch Template

# Dependencies
- (For docker) rootless docker

# Using docker
  1. Change variables in `docker-compose.yml` as needed
  1. Add scripts into `./docker/Dockerfile` as needed
  1. Perform the following command for building your docker image

      ```
        docker compose build
      ```
  1. Run a docker container via the following command 

      ```
        docker compose up -d
      ```
  1. Run bash inside of the container
      
      ```
        docker compose exec <project_name> /bin/bash
      ```
  1. Install dependent packages by Poetry (This process only needs to be done once)
  
      ```
        poetry install
      ```
  Note that all changes in the docker container will be deleted when you exit docker container. 

# Running Jupyter
  1. Run a docker container
  
    ```
      docker compose up -d
    ```
  1. Run bash inside of the container

    ```
      docker compose exec <project_name> /bin/bash
    ```
  1. Run `docker/run_jupyter.sh`

    ```
      ./docker/run_jupyter.sh
    ```
  1. When you write and run your scripts, you need to choose the kernel *project_name*

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

