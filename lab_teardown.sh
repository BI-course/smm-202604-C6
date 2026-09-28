#!/bin/bash

set -euo pipefail

# Resolve script location -- run docker compose commands from here so
# relative paths in the compose files (./model, ./frontend, etc.) resolve
# the same way they did when you first ran "docker compose up".
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="$SCRIPT_DIR/container-volumes"
NGINX_CERTS_DIR="$BASE_DIR/nginx/certs"
UBUNTU_HOME_DIR="$BASE_DIR/ubuntu/home-student"

# -----------------------------------------------------------------------
# Usage:
#   ./project_teardown.sh [OVERLAY_FILE] [-y|--yes]
#
#   OVERLAY_FILE  Optional. Pass the SAME overlay you used with "up", e.g.
#                 "docker-compose-dev.yaml" or "docker-compose-prod.yaml".
#                 If you started the stack with:
#                     docker compose -f docker-compose.yaml -f docker-compose-dev.yaml up
#                 you MUST tear it down with the same combination, or some
#                 containers/networks/volumes defined only in the overlay
#                 file will be left running or left behind.
#
#   -y, --yes     Skip the confirmation prompt before deleting the local
#                 container-volumes/ folder (useful for CI, not needed for
#                 normal classroom use).
# -----------------------------------------------------------------------

OVERLAY_FILE="" # Leave this empty. Pass the overlay filename as a command-line
                # argument instead (see the Usage note above) -- do not hardcode
                # a filename or a command string here, or the script will try
                # to use it on every run, even when you meant to tear down the
                # base docker-compose.yaml alone.
SKIP_CONFIRM="false"

for arg in "$@"; do
    case "$arg" in
        -y|--yes)
            SKIP_CONFIRM="true"
            ;;
        *)
            OVERLAY_FILE="$arg"
            ;;
    esac
done

COMPOSE_FILES=(-f docker-compose.yaml)
if [ -n "$OVERLAY_FILE" ]; then
    if [ ! -f "$OVERLAY_FILE" ]; then
        echo "Error: overlay file '$OVERLAY_FILE' not found in $SCRIPT_DIR. Aborting."
        exit 1
    fi
    COMPOSE_FILES+=(-f "$OVERLAY_FILE")
    echo "Tearing down using: docker-compose.yaml + $OVERLAY_FILE"
else
    echo "Tearing down using: docker-compose.yaml only"
    echo "(Pass the overlay filename as an argument if you started the stack with one, e.g. ./project_teardown.sh docker-compose-dev.yaml)"
fi

# -----------------------------------------------------------------------
# Step 1: Stop containers and remove networks, named volumes, and the
# images this project itself built (--rmi all also removes the base
# images like python:3.12.10-slim and nginx:1.28.1 that were pulled for
# this project, so this IS a full teardown -- the next "docker compose up"
# will re-download and re-build everything from scratch).
# -----------------------------------------------------------------------
echo "Stopping containers and removing networks, named volumes, and images..."
docker compose "${COMPOSE_FILES[@]}" down -v --rmi all

# -----------------------------------------------------------------------
# Step 2: Empty out the two bind-mounted folders under container-volumes/,
# WITHOUT deleting the container-volumes/ tree itself or the two folders
# themselves (so the layout setup.sh created is still there, ready for the
# next "docker compose up" with nothing stale left inside). This is
# destructive to their CONTENTS -- anything a student saved inside the
# "ubuntu" container's home directory, or the TLS certs generated for
# nginx, is gone after this. Confirm before deleting, unless -y/--yes
# was passed.
# -----------------------------------------------------------------------
empty_dir_contents() {
    local dir="$1"
    if [ -d "$dir" ]; then
        # -mindepth 1 skips the directory itself, so only its CONTENTS
        # (including hidden files/subfolders) are deleted, not the folder.
        find "$dir" -mindepth 1 -delete
        echo "Emptied: $dir"
    else
        echo "Not found (nothing to empty): $dir"
    fi
}

if [ -d "$NGINX_CERTS_DIR" ] || [ -d "$UBUNTU_HOME_DIR" ]; then
    PROCEED="true"
    if [ "$SKIP_CONFIRM" != "true" ]; then
        echo "This will delete the CONTENTS of:"
        echo "  $NGINX_CERTS_DIR"
        echo "  $UBUNTU_HOME_DIR"
        echo "(The folders themselves are kept -- only what's inside them is removed.)"
        read -r -p "Proceed? [y/N] " REPLY
        echo
        if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            echo "Skipping cleanup of container-volumes contents."
            PROCEED="false"
        fi
    fi

    if [ "$PROCEED" = "true" ]; then
        empty_dir_contents "$NGINX_CERTS_DIR"
        empty_dir_contents "$UBUNTU_HOME_DIR"
    fi
else
    echo "Neither $NGINX_CERTS_DIR nor $UBUNTU_HOME_DIR was found -- nothing to empty."
fi

echo "Done. Local Docker resources for this project have been torn down."
echo "Reminder: .env and .venv are not touched by this script -- remove them"
echo "yourself once you no longer need them (see cleanup_instructions.md)."
