#!/usr/bin/env python3
"""Build the package as pip would, install the wheel, and run tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import venv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = ROOT / ".artifacts" / "wheel-install-test"
DEFAULT_DIST = DEFAULT_WORKDIR / "dist"


def run(
    cmd: list[str | Path],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


def pip_index_args(torch_index_url: str | None) -> list[str]:
    if not torch_index_url:
        return []
    return [
        "--index-url",
        torch_index_url,
        "--extra-index-url",
        "https://pypi.org/simple",
    ]


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_env(venv_dir: Path, base_env: dict[str, str]) -> dict[str, str]:
    env = base_env.copy()
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    return env


def create_venv(venv_dir: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    return venv_python(venv_dir)


def latest_wheel(dist: Path) -> Path:
    wheels = sorted(dist.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"No wheel was built in {dist}")
    return wheels[-1]


def clean_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_requirements() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)
    return list(pyproject["build-system"]["requires"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="Scratch directory for temporary virtualenvs.",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=DEFAULT_DIST,
        help="Directory where the built wheel should be written.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help=(
            "Keep prior workdir contents instead of starting from a clean "
            "scratch area."
        ),
    )
    parser.add_argument(
        "--preinstall-build-deps",
        action="store_true",
        help=(
            "Install build-system requirements into the build venv and disable "
            "pip build isolation."
        ),
    )
    parser.add_argument(
        "--torch-index-url",
        default=None,
        help=(
            "Optional PyTorch package index URL. CI uses the CPU index to avoid "
            "downloading CUDA wheels on hosted runners."
        ),
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional pytest arguments, for example: -- tests/test_loading.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    dist = args.dist.resolve()

    if not args.keep:
        clean_path(workdir)
        clean_path(dist)

    workdir.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LIBTORCH_USE_PYTORCH"] = "1"

    print(
        textwrap.dedent(
            f"""
            Platform: {platform.platform()}
            Driver Python: {sys.executable}
            Workdir: {workdir}
            Dist: {dist}
            """
        ).strip(),
        flush=True,
    )

    build_venv = workdir / "build-venv"
    build_python = create_venv(build_venv)
    build_env = venv_env(build_venv, env)
    index_args = pip_index_args(args.torch_index_url)
    if args.preinstall_build_deps:
        run(
            [
                build_python,
                "-m",
                "pip",
                "install",
                *index_args,
                *build_requirements(),
            ],
            env=build_env,
        )

    build_command = [
        build_python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        dist,
    ]
    if args.preinstall_build_deps:
        build_command.append("--no-build-isolation")
    build_command.append(ROOT)
    run(
        build_command,
        env=build_env,
    )

    wheel = latest_wheel(dist)
    print(f"Built wheel: {wheel.name}", flush=True)

    test_venv = workdir / "test-venv"
    test_python = create_venv(test_venv)
    test_env = venv_env(test_venv, env)
    run(
        [
            test_python,
            "-m",
            "pip",
            "install",
            *index_args,
            wheel,
            "pytest>=8.3",
        ],
        env=test_env,
    )
    run([test_python, "-m", "pip", "check"], env=test_env)

    smoke = r"""
import importlib.metadata as metadata
from pathlib import Path

import torch
import tdataframe
from tdataframe._torch import ensure_torch_loaded

ensure_torch_loaded()
import tdataframe._internal as internal
from tdataframe.alignment import with_barycenters
from tdataframe.loading import GraphMl, Stl, WeightedObj

torch_version = torch.__version__.split("+", 1)[0]
if torch_version != "2.7.0":
    raise SystemExit(f"Expected torch 2.7.0 for tch 0.20.0, got {torch.__version__}")

print("TDAtaframe", metadata.version("TDAtaframe"), Path(tdataframe.__file__).parent)
print("Rust extension", internal.__version__)
print("Torch", torch.__version__, Path(torch.__file__).parent)
print(
    "Smoke imports",
    with_barycenters.__name__,
    GraphMl.__name__,
    Stl.__name__,
    WeightedObj.__name__,
)
"""
    with tempfile.TemporaryDirectory() as tmp:
        run([test_python, "-c", smoke], cwd=Path(tmp), env=test_env)

    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["tests"]

    run([test_python, "-m", "pytest", *pytest_args], env=test_env)


if __name__ == "__main__":
    main()
