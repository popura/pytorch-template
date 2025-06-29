#!/bin/bash

uv run ipython kernel install --user --name ${PROJECT_NAME}
uv run --with jupyter jupyter notebook --allow-root --ip=${JUPYTER_IP} --port=${JUPYTER_PORT} --no-browser --notebook-dir=${JUPYTER_NOTEBOOK_DIR}
