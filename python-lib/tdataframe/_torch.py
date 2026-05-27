"""Runtime helpers for loading the libtorch shipped with PyTorch."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_DLL_DIRECTORIES = []
_TORCH_LOADED = False


def ensure_torch_loaded() -> None:
    """Import torch and make its shared libraries visible to plugin loaders."""
    global _TORCH_LOADED
    if _TORCH_LOADED:
        return

    import torch

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if sys.platform == "win32" and torch_lib.exists():
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(torch_lib)))

    _TORCH_LOADED = True
