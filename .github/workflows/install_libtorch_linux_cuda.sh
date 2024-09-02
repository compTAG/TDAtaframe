#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download libtorch
curl -L https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip -o libtorch.zip


# Unzip the libraries to /usr/lib
sudo unzip libtorch.zip

cd sudo cp libtorch/lib/* /usr/lib/

# Clean up the zip file
rm libtorch.zip

# Update the linker cache
sudo ldconfig