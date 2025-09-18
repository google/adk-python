"""
Backend module for GCP Security Agent
"""

from pathlib import Path

# Ensure the cache directory exists
cache_dir = Path(__file__).parent / "cache"
cache_dir.mkdir(exist_ok=True)

__all__ = ["main"]