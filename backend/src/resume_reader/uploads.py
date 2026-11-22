"""Read an uploaded file in bounded pieces."""

from __future__ import annotations

from fastapi import UploadFile

# The route stops once the buffered file passes the configured limit.
UPLOAD_PIECE_BYTES = 64 * 1024


class UploadTooLarge(Exception):
    """The file grew past the configured limit while it was being read."""


async def read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read ``upload`` in 64 KiB pieces.

    The returned bytes are the whole file when it fits in ``limit``. A larger
    file raises ``UploadTooLarge`` and is not read any further.
    """
    pieces: list[bytes] = []
    total = 0
    try:
        while True:
            piece = await upload.read(UPLOAD_PIECE_BYTES)
            if not piece:
                break
            total += len(piece)
            if total > limit:
                raise UploadTooLarge
            pieces.append(piece)
        return b"".join(pieces)
    finally:
        await upload.close()
