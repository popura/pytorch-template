# ---------- 1st stage ----------
ARG BASE_IMAGE=ubuntu:22.04
ARG PROJECT_NAME=pytorch-template
ARG UV_TORCH_SOURCE=cpu
ARG APPLICATION_DIRECTORY=/${PROJECT_NAME}
ARG UV_VERSION=0.7.12
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

# ---------- 2nd stage ----------
FROM ${BASE_IMAGE}
ARG BASE_IMAGE=ubuntu:22.04
ARG PROJECT_NAME=pytorch-template
ARG APPLICATION_DIRECTORY=/${PROJECT_NAME}
ARG UV_TORCH_SOURCE=cpu
ENV HOME=/root
ENV UV_PROJECT_ENVIRONMENT=${HOME}/.venv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
LABEL maintainer="Yuma Kinoshita"

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
COPY --from=uv-bin /uv /uvx /bin/

ADD . ${APPLICATION_DIRECTORY}
WORKDIR ${APPLICATION_DIRECTORY}

RUN --mount=type=cache,target=${HOME}/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv ${UV_PROJECT_ENVIRONMENT} && \
    uv sync --frozen --no-install-project --extra ${UV_TORCH_SOURCE}

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:$PATH"
