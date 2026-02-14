"""Utilities for normalizing A2A public URLs and mount paths."""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_path(path: str) -> str:
  """Normalize an application path to a canonical mount path."""
  path = (path or "/").strip()
  if not path:
    return "/"
  if not path.startswith("/"):
    path = f"/{path}"
  if path != "/":
    path = path.rstrip("/")
  return path


def normalize_public_url(url: str) -> str:
  """Normalize a public URL and validate required URL components."""
  parsed = urlparse(url)
  if not parsed.scheme or not parsed.netloc:
    raise ValueError(
        "http_url must include a scheme and host, for example "
        "'https://example.com/analysis-agent'."
    )
  normalized_path = normalize_path(parsed.path)
  if normalized_path == "/":
    return f"{parsed.scheme}://{parsed.netloc}"
  return f"{parsed.scheme}://{parsed.netloc}{normalized_path}"


def build_public_url(protocol: str, host: str, port: int, path: str) -> str:
  """Build a normalized public URL from host, port, protocol and path."""
  normalized_path = normalize_path(path)
  base = f"{protocol}://{host}:{port}"
  if normalized_path == "/":
    return base
  return f"{base}{normalized_path}"
