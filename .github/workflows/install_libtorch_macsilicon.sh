
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download libtorch
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip -o libtorch.zip


# Unzip the libraries to a writable location within the current workspace
unzip libtorch.zip -d $(pwd)

# Clean up the zip file
rm libtorch.zip

export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTORCH/lib
