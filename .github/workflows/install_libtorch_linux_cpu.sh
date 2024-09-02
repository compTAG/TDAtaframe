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
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip -o libtorch.zip

# Unzip the libraries to a writable location within the current workspace
unzip libtorch.zip -d $(pwd)

# Clean up the zip file
rm libtorch.zip

export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTORCH/lib



