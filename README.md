# TDAtaframe
TDA brought to dataframes.

# Install
TDAtaframe is available on pypi as a source distribution. 
The backend requires that you have the appropriate libtorch 2.5.0 libraries 
installed on your system in the standard location.
After ensuring that these libraries are present, all that is needed is
```pip install tdataframe```

## Manual libtorch install instructions
Note that symlinking the libraries will not work. The appropriate libraries
and headers need to be present under /usr/lib and /usr/include respectively
for tch-rs to find the system libraries without setting environment variables.
Download libtorch from [here](https://pytorch.org/).
`unzip libtorch-*.zip && cd libtorch`

## Alternative libtorch locations
todo. see tch-rs docs
