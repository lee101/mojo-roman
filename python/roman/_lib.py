"""Load and call the compiled Mojo conversion library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "roman.mojo")
LIB = os.environ.get("MOJO_ROMAN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-roman.so"
)

I = ctypes.c_int64
_SIGNATURES = {
    "mr_to_roman": ([I, I, I], I),
    "mr_from_roman": ([I, I, I, I], I),
    "mr_to_roman_batch": ([I, I, I, I, I], I),
    "mr_from_roman_batch": ([I, I, I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_ROMAN_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(SRC):
            return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside pixi or set MOJO_ROMAN_LIB")
    proc = subprocess.run(
        [os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        details = "\n".join(
            output.strip() for output in (proc.stdout, proc.stderr) if output.strip()
        )
        raise BuildError(details[:4000] or "Mojo build failed without output")
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def bytes_addr(data: bytes) -> int:
    pointer = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
    if pointer is None:
        raise ValueError("cannot take the address of an empty byte string")
    return pointer
