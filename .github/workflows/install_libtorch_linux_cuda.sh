#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to install unzip based on available package manager
install_unzip() {
    if command -v apt-get &> /dev/null; then
        echo "Using apt-get to install unzip"
        apt-get update
        apt-get install -y unzip
    elif command -v yum &> /dev/null; then
        echo "Using yum to install unzip"
        yum install -y unzip
    elif command -v apk &> /dev/null; then
        echo "Using apk to install unzip"
        apk add unzip
    else
        echo "No compatible package manager found. Please install 'unzip' manually."
        exit 1
    fi
}

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, installing..."
    install_unzip
fi
# Download libtorch
curl -L https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip -o libtorch.zip

# Unzip the libraries to /opt/libtorch
unzip libtorch.zip -d /opt/

export LIBTORCH=/opt/libtorch

# Clean up the zip file
rm libtorch.zip

