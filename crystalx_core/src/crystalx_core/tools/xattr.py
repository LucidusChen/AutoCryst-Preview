"""Read-only extended-attribute checks without third-party packages.

Python's ``os.getxattr`` exists only on Linux. On macOS -- where the attribute that matters, the
Gatekeeper quarantine flag ``com.apple.quarantine``, lives -- it is absent, so a check built on it
silently reported "not quarantined" for every file. Here the macOS path calls libc's
``getxattr(2)`` through :mod:`ctypes`. Nothing is ever written or removed: clearing quarantine is a
security decision for the owner of the machine.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

__all__ = ["has_xattr", "QUARANTINE_XATTR"]

QUARANTINE_XATTR = "com.apple.quarantine"

#: macOS ``ENOATTR``; Python's errno module defines it there, the literal is a fallback.
_DARWIN_ENOATTR = getattr(errno, "ENOATTR", 93)


def has_xattr(path: Path | str, name: str) -> bool | None:
    """``True`` if ``name`` is set on ``path``, ``False`` if not, ``None`` if this cannot be told."""
    if sys.platform == "darwin":
        return _darwin_has_xattr(os.fsencode(str(path)), name.encode("utf-8"))
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return None
    try:
        getxattr(str(path), name)
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
            # Absent, or the name/filesystem cannot carry it -- either way it is not set.
            return False
        return None
    return True


def _darwin_has_xattr(path: bytes, name: bytes) -> bool | None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getxattr = libc.getxattr
    except (OSError, AttributeError):
        return None
    getxattr.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    getxattr.restype = ctypes.c_ssize_t
    size = getxattr(path, name, None, 0, 0, 0)
    if size >= 0:
        return True
    if ctypes.get_errno() == _DARWIN_ENOATTR:
        return False
    return None
