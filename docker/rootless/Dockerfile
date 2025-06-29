ARG BASE_IMAGE=ubuntu22.04
FROM ${BASE_IMAGE}
LABEL maintainer="Yuma Kinoshita"

ARG PROJECT_NAME=pytorch-template
ARG UV_VERSION=0.7.12
ARG UV_TORCH_SOURCE=cpu
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ARG PYTHON_VERSION=3.13.3
ARG APPLICATION_DIRECTORY=/${PROJECT_NAME}

ENV HOME=/root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# --- Change repository ---
# RUN sed -i.org -e 's|archive.ubuntu.com|ubuntutym.u-toyama.ac.jp|g' /etc/apt/sources.list && \
#     apt-get clean

# --- Install Packages with Apt ---
RUN apt-get update -y && apt-get upgrade -y --fix-missing && \
    apt-get install -y --fix-missing \
    build-essential \
    zip unzip curl wget vim tree graphviz && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

# --- Install uv ---
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /${APPCILATION_DIRECTORY}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra ${cpu} --frozen

WORKDIR ${APPLICATION_DIRECTORY}
ENV PATH="/${APPLICATION_DIRECTORY}/.venv/bin:$PATH"
