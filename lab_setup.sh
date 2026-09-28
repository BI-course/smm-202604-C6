#!/bin/bash

set -euo pipefail

# Resolve script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/container-volumes"

echo "Creating volume directories..."

# Prevent accidental overwrite of unexpected file
if [ -e "$BASE_DIR" ] && [ ! -d "$BASE_DIR" ]; then
    echo "Error: $BASE_DIR exists but is not a directory. Aborting."
    exit 1
fi

if [ -L "$BASE_DIR" ]; then
    echo "Error: $BASE_DIR is a symbolic link. Aborting."
    exit 1
fi

# Optional: warn if reusing existing data
if [ -d "$BASE_DIR" ] && compgen -A file "$BASE_DIR" > /dev/null; then
    echo "Warning: Existing data detected in $BASE_DIR"
fi

# Create directories
# - ubuntu/home-student  : bind-mounted into the "ubuntu" (customized-ubuntu-server-smm)
#                          container as the student's home directory, so files
#                          survive a "docker compose down" and are visible from
#                          the host at the same time.
# - nginx/certs          : bind-mounted into the "nginx" container so the
#                          self-signed TLS certificate/key you generate
#                          (cert.pem / key.pem) are readable by nginx and
#                          persist across container restarts/rebuilds.
mkdir -p \
    "$BASE_DIR/ubuntu/home-student" \
    "$BASE_DIR/nginx/certs"

echo "Setting permissions..."

# Safe permission handling
chmod -R 775 "$BASE_DIR"

echo "Done. You may now proceed to the next step."
