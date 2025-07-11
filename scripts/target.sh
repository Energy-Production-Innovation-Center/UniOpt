#!/usr/bin/env bash
# This script sets up the target environment in which the app will be built.

# Exit on error.
set -eo pipefail

# Update repositories.
dnf -y update

packages=(
    # python deps
    python3.12
    python3.12-pip
)

# Install packages.
dnf -y install "${packages[@]}"

# Cleanup.
dnf -y clean all
rm -rf /var/cache

# Aliases.
ln -s /usr/bin/python3.12 /usr/bin/python
ln -s /usr/bin/pip3.12 /usr/bin/pip
