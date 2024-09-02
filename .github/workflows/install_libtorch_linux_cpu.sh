#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install unzip if not available
if ! command -v unzip &> /dev/null
then
    echo "unzip could not be found, installing..."
    yum install -y unzip
fi

# Download libtorch
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip -o libtorch.zip

# Unzip the libraries to /opt/libtorch
unzip libtorch.zip -d /opt/

export LIBTORCH=/opt/libtorch

# Clean up the zip file
rm libtorch.zip
