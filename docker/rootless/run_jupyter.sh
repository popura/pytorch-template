#!/bin/bash

poetry run python -m ipykernel install --name ${PROJECT_NAME} --user
jupyter notebook --allow-root --ip=${JUPYTER_IP} --port=${JUPYTER_PORT} --no-browser --notebook-dir=${JUPYTER_NOTEBOOK_DIR}
