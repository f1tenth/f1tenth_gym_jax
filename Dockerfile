FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1 \
    PATH="/f1tenth_gym_jax/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libegl1 \
        libgl1 \
        libxkbcommon-x11-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /f1tenth_gym_jax
COPY . /f1tenth_gym_jax

RUN python -m pip install --upgrade pip uv && \
    uv sync --frozen --no-dev

ENTRYPOINT ["/bin/bash"]
