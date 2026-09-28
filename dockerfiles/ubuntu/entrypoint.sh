#!/bin/bash
set -e

INIT_MARKER="/var/lib/container-initialized"

# First boot only
if [ ! -f "$INIT_MARKER" ]; then
    if [ -z "$ROOT_PASSWORD" ] || [ -z "$STUDENT_PASSWORD" ]; then
        echo "ERROR: ROOT_PASSWORD and STUDENT_PASSWORD must be set on first boot"
        exit 1
    fi

    echo "Initializing container (first boot)..."

    echo "root:${ROOT_PASSWORD}" | chpasswd
    echo "student:${STUDENT_PASSWORD}" | chpasswd

    mkdir -p /var/lib
    touch "$INIT_MARKER"

    echo "Initialization complete."
else
    echo "Container already initialized. Skipping password setup."
fi

exec "$@"
