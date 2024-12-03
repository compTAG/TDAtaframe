# TDAtaframe
TDA brought to dataframes.

# Install
TDAtaframe is available on pypi as a source distribution.
The backend requires that you have the appropriate libtorch 2.5.1 libraries 
installed on your system in the standard location.

On Linux/macOS, this is typically /usr/lib, /usr/include, /usr/share for 
library, include, and share files respectively. Sym-linking the files will not 
work.

After ensuring that these libraries are present, all that is needed is

```pip install tdataframe```.

On first install, it is normal for compilation to take a few minutes.

## Manual libtorch install instructions
It is recommended to use your package manager to install libtorch.
If not readily available for your system, follow the below instructions 
as a workaround.

Download libtorch from [here](https://pytorch.org/).
Then, unzip the files.

`unzip libtorch-*.zip && cd libtorch`

Next, copy all the files to the system. Make sure
you understand what these commands do as they are difficult to undo.
You will likely need to run these commands with `sudo`.

`cp -r lib/* /usr/lib && cp -r include/* /usr/include && cp -r share /usr/share`

Finally, update your linker cache with

`ldconfig`.

## Alternative libtorch locations
todo. see tch-rs docs
