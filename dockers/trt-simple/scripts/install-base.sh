#!/bin/sh
set -eu

/opt/trt-simple/scripts/apt_with_proxy.sh sh -c '
  apt-get update &&
  apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    wget \
    gnupg2 \
    python3 \
    python3-pip &&
  rm -rf /var/lib/apt/lists/*
'

