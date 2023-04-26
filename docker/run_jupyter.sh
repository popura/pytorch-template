#!/bin/bash

poetry run python -m ipykernel install --name ${PROJECT_NAME} --user
jupyter notebook --allow-root
