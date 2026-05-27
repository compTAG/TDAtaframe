# TDAtaframe
TDA brought to dataframes.

# Install
TDAtaframe uses the libtorch libraries provided by the Python `torch` package.
You do not need to install libtorch system-wide.

Install with

```bash
pip install tdataframe
```

The Rust backend currently targets `torch==2.7.0`, matching `tch-rs 0.20.0`.
If pip builds from source, it will compile against the `torch` package in the
build environment.

On first install, it is normal for compilation to take a few minutes.
