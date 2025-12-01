#!/bin/sh
set -eu

/opt/trt-simple/scripts/apt_with_proxy.sh sh -c '
  apt-get update &&
  apt-get install -y --no-install-recommends \
    tensorrt \
    python3-libnvinfer-dev &&
  rm -rf /var/lib/apt/lists/*
'

