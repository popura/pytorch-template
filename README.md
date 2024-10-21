# PyTorch Template

# Dependencies
- (For docker) rootless/rootful docker

# Using rootless docker
  1. Change variables in `./docker/docker-compose.yml` as needed.
  Especially, you may substitute `pytorch-template` in the file with your project name.
  1. Add scripts into `./docker/rootless/Dockerfile` as needed.
  It will works without any changes.
  1. Change variables in `pyproject.toml` as needed.
  Especially, you may change `name` and `authors` in the file.
  1. To build your Docker image, run the following command in the directory containing the `docker-compose.yml` file:

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
  1. Run the following command to verify that your installation was successful.
  
      ```
        poetry run python scripts/installation_verification.py
      ```

  Note that all changes in the docker container will be deleted when you exit docker container. 

### GPU Support
To enable GPU calculations, you can override `docker-compose.yml` with `docker-compose.gpu.yml`
by creating `./docker/.env` file as follows:

  ```
    echo COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml >> ./docker/.env
  ```

### Rootful docker
If you're unable to use rootless Docker,
you'll need to override the default settings with `docker-compose.rootful.yml`.
Also, you may add scripts into `./docker/rootful/Dockerfile`.

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


# Developing your package
  1. For prototyping, make `notebooks/local` directory and add jupyter notebook into the directory.
  1. For developing, add source files into `src` directory
  1. For testing, add test codes into `test` directory
  1. For evaluation, add script files into `scripts` directory

