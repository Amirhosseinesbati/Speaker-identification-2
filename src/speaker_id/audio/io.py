"""Content-aware audio decoding and read-only container integrity inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import soundfile as sf


def open_audio(path: str | Path) -> sf.SoundFile:
    """Let libsndfile inspect content; never force a codec from the extension."""
    return sf.SoundFile(str(path), mode="r")


def file_sha256(path: str | Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def inspect_wave_payload(path: str | Path) -> dict:
    """Compare independent RIFF declarations with physical file boundaries.

    MP3 bitstream completeness cannot be established from libsndfile decoding;
    for non-RIFF/WAVE formats this check explicitly returns ``not_checked``.
    Chunk payloads are skipped without allocating or modifying audio data.
    """
    path = Path(path)
    size = path.stat().st_size
    result = {"wave_integrity": "not_checked", "wave_payload_frames": None,
              "wave_integrity_error": ""}
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return result
        errors = []
        if struct.unpack("<I", header[4:8])[0] + 8 != size:
            errors.append("RIFF declared length differs from physical size")
        block_align = None
        data_bytes = 0
        data_chunks = 0
        position = 12
        while position + 8 <= size:
            stream.seek(position)
            chunk = stream.read(8)
            length = struct.unpack("<I", chunk[4:])[0]
            end = position + 8 + length
            if end > size:
                errors.append("chunk extends beyond physical EOF")
            if chunk[:4] == b"fmt " and length >= 16:
                fmt = stream.read(16)
                if len(fmt) == 16:
                    block_align = struct.unpack("<H", fmt[12:14])[0]
            elif chunk[:4] == b"data":
                data_chunks += 1
                data_bytes += length
            if end > size:
                break
            position = end + length % 2
        if data_chunks != 1:
            errors.append(f"expected one data chunk, found {data_chunks}")
        if not block_align:
            errors.append("missing or invalid block alignment")
        elif data_bytes % block_align:
            errors.append("declared payload has partial audio frame")
        else:
            result["wave_payload_frames"] = data_bytes // block_align
        if position < size and position + 8 > size:
            errors.append("trailing bytes cannot form a RIFF chunk")
        result["wave_integrity"] = "error" if errors else "ok"
        result["wave_integrity_error"] = "; ".join(errors)
    return result
