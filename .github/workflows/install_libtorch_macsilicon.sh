
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download libtorch
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip -o libtorch.zip

# Unzip the libraries to /opt/libtorch
unzip libtorch.zip -d /opt/

export LIBTORCH=/opt/libtorch

# Clean up the zip file
rm libtorch.zip
