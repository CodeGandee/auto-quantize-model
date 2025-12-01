#!/bin/sh
set -eu

# Normalize proxy environment for use inside Docker during build.
# If the proxy points at 127.0.0.1 or localhost, rewrite it to
# host.docker.internal so it can reach the host from the container.
PROXY="${http_proxy:-${HTTP_PROXY:-}}"

if [ -n "${PROXY}" ]; then
  PROXY=$(printf '%s' "${PROXY}" \
    | sed 's/127\.0\.0\.1/host.docker.internal/g; s/localhost/host.docker.internal/g')
  export http_proxy="${PROXY}" https_proxy="${PROXY}" HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}"
fi

exec "$@"

