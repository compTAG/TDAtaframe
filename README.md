# TDAtaframe
TDA brought to dataframes.

## Documentation

Repo-level docs live in [docs/README.md](docs/README.md).

- [Architecture](docs/architecture.md)
- [Data Model](docs/data-model.md)
- [Development Guide](docs/development.md)

# Install
TDAtaframe uses the libtorch libraries provided by the Python `torch` package.
You do not need to install libtorch system-wide.

Install with

```bash
pip install tdataframe
```

The Rust backend currently targets `torch==2.7.0`, matching `tch-rs 0.20.0`.
The package supports Python 3.12 and 3.13. If pip builds from source, it will
compile against the `torch` package in the build environment.

On first install, it is normal for compilation to take a few minutes.

# Verify Packaging
Run the same install path that CI uses:

```bash
python3 scripts/wheel_install_test.py
```

That script builds a wheel through pip's isolated PEP 517 path, installs the
wheel into a fresh virtualenv, runs `pip check`, imports the Rust extension
against PyTorch's bundled libtorch, and then runs the test suite from the
installed wheel.

For cross-machine checks, GitHub Actions runs the wheel test on Linux x64,
macOS x64, macOS arm64, and Windows x64. If you have the GitHub CLI installed
and authenticated, this command triggers CI and saves the logs locally:

```bash
python3 scripts/ci_watch.py
```
