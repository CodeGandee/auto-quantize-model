FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG USERNAME=me
ARG UID=3100
ARG GID=1100
ARG http_proxy
ARG HTTP_PROXY

ENV DEBIAN_FRONTEND=noninteractive

# Copy helper scripts into the image for proxy-aware apt operations.
COPY dockers/trt-simple/scripts /opt/trt-simple/scripts
RUN chmod +x /opt/trt-simple/scripts/*.sh

# Base tools and Python (proxy-aware via helper script).
RUN /opt/trt-simple/scripts/install-base.sh

# Create group/user matching host UID/GID and grant sudo (passwordless).
RUN if ! getent group "${GID}" >/dev/null; then \
        groupadd -g "${GID}" "${USERNAME}"; \
    else \
        groupmod -n "${USERNAME}" "$(getent group "${GID}" | cut -d: -f1)"; \
    fi && \
    if ! id -u "${UID}" >/dev/null 2>&1; then \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    else \
        usermod -d "/home/${USERNAME}" -l "${USERNAME}" "$(getent passwd "${UID}" | cut -d: -f1)" || true; \
    fi && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-${USERNAME} && \
    chmod 0440 /etc/sudoers.d/99-${USERNAME}

# Install TensorRT via Debian packages from the CUDA repo already configured
# in the base image (proxy-aware via helper script).
RUN /opt/trt-simple/scripts/install-trt.sh

# Minimal TensorRT sanity check: import from Python.
RUN python3 -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)"

USER ${USERNAME}
WORKDIR /workspace
