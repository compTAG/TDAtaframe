#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download libtorch
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip -o libtorch.zip

# Unzip the libraries to /usr/lib
sudo unzip libtorch.zip -d /usr/lib

# Clean up the zip file
rm libtorch.zip

# Update the linker cache
sudo ldconfig
