"""Test helpers (the chapter 1 synthetic-unit helpers moved to legacy/tests/helpers.py)."""

from __future__ import annotations


def free_addr() -> str:
    """A free TCP port on 127.0.0.1 chosen by the OS (tests never use fixed ports, so they cannot clash with jobs)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp:127.0.0.1:{s.getsockname()[1]}"
