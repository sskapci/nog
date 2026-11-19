"""Message framing between gateway and hosts: u32 length, u8 kind, payload; TCP ("tcp:HOST:PORT") or Unix sockets
("unix:PATH").

Moved unchanged from mxn/host/node.py (cleanup 2026-10-30). Security: frames are not authenticated or encrypted.
Use this only inside a trusted LAN, with the ports (7100-7102, 7199) open between A and B and closed to everything
else.
"""

from __future__ import annotations

import asyncio
import struct


def frame(kind: int, payload: bytes) -> bytes:
    return struct.pack("<IB", len(payload) + 1, kind) + payload


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    n, kind = struct.unpack("<IB", await reader.readexactly(5))
    return kind, await reader.readexactly(n - 1)


async def open_conn(addr: str):
    kind, _, where = addr.partition(":")
    if kind == "unix":
        return await asyncio.open_unix_connection(where)
    host, _, port = where.rpartition(":")
    return await asyncio.open_connection(host, int(port))


async def serve(addr: str, handler):
    kind, _, where = addr.partition(":")
    if kind == "unix":
        return await asyncio.start_unix_server(handler, where)
    host, _, port = where.rpartition(":")
    return await asyncio.start_server(handler, host, int(port))
