"""Prevent credentials from entering tracking metadata or source archives."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(
    r"(?:password|passwd|token|secret|authorization|credential|private[_-]?key|"
    r"api[_-]?key|access[_-]?key|client[_-]?key|cookie)", re.IGNORECASE
)
_INLINE_SECRET = re.compile(
    r"(?i)\b(authorization|password|passwd|token|secret|api[_-]?key|access[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def safe_endpoint(uri: str) -> str:
    """Keep endpoint identity, never URL user-info, queries, or fragments."""
    parsed = urlsplit(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A remote HTTP(S) MLflow tracking URI is required.")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme.lower(), host + port, parsed.path.rstrip("/"), "", ""))


class Redactor:
    def __init__(self, environ: Mapping[str, str] | None = None, extra_values=()):
        environment = os.environ if environ is None else environ
        values = [str(v) for k, v in environment.items() if _SECRET_KEY.search(k) and v]
        values.extend(str(v) for v in extra_values if v)
        self.secret_values = tuple(sorted(set(values), key=len, reverse=True))

    @staticmethod
    def secret_key(key: str) -> bool:
        return bool(_SECRET_KEY.search(key))

    def text(self, value: str) -> str:
        for secret in self.secret_values:
            value = value.replace(secret, REDACTED)

        def clean_url(match):
            try:
                parsed = urlsplit(match.group(0))
                host = parsed.hostname or ""
                if ":" in host:
                    host = f"[{host}]"
                port = f":{parsed.port}" if parsed.port is not None else ""
                query = urlencode([
                    (key, REDACTED if self.secret_key(key) else val)
                    for key, val in parse_qsl(parsed.query, keep_blank_values=True)
                ])
                return urlunsplit((parsed.scheme, host + port, parsed.path, query, ""))
            except ValueError:
                return "[REDACTED_URL]"

        value = _URL.sub(clean_url, value)
        return _INLINE_SECRET.sub(lambda m: m.group(1) + m.group(2) + REDACTED, value)

    def __call__(self, value):
        if isinstance(value, Mapping):
            return {
                self.text(str(key)): REDACTED if self.secret_key(str(key)) else self(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [self(item) for item in value]
        if isinstance(value, (str, Path)):
            return self.text(str(value))
        if value is None or isinstance(value, (bool, int, float)):
            return value
        raise TypeError(f"Unsupported tracking metadata type: {type(value).__name__}")

    def assert_no_secret_bytes(self, payload: bytes, label: str) -> None:
        # Archives must preserve exact source bytes. Refuse secret-bearing files,
        # instead of silently modifying executable code while archiving it.
        for secret in self.secret_values:
            if secret.encode("utf-8") in payload:
                raise ValueError(f"Refusing to archive a credential value in {label}.")
