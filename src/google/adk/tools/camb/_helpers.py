# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for CAMB AI tools.

Provides lazy client management, polling, audio format detection,
PCM-to-WAV conversion, and result formatting utilities.
"""

from __future__ import annotations

import asyncio
import json
import struct
import tempfile
from typing import Any
from typing import Optional


class CambHelpers:
  """Shared helper utilities for CAMB AI tool functions.

  Manages a lazily-initialised ``AsyncCambAI`` client and provides
  common helper methods used across all CAMB tool functions.

  Args:
    api_key: camb.ai API key.  Falls back to ``CAMB_API_KEY`` env var.
    timeout: Request timeout in seconds.
    max_poll_attempts: Maximum polling iterations for async tasks.
    poll_interval: Seconds between polling attempts.
  """

  def __init__(
      self,
      api_key: Optional[str] = None,
      timeout: float = 60.0,
      max_poll_attempts: int = 60,
      poll_interval: float = 2.0,
  ) -> None:
    from os import getenv

    self._api_key = api_key or getenv("CAMB_API_KEY")
    if not self._api_key:
      raise ValueError(
          "CAMB_API_KEY not set. Please set the CAMB_API_KEY environment"
          " variable or pass api_key to CambAIToolset."
      )
    self._timeout = timeout
    self._max_poll_attempts = max_poll_attempts
    self._poll_interval = poll_interval
    self._client: Any = None

  # ------------------------------------------------------------------
  # Client
  # ------------------------------------------------------------------

  def get_client(self) -> Any:
    """Return a lazily-initialised ``AsyncCambAI`` client."""
    if self._client is None:
      try:
        from camb.client import AsyncCambAI
      except ImportError as exc:
        raise ImportError(
            "The 'camb' package is required to use CambAIToolset. "
            "Install it with: pip install google-adk[camb]"
        ) from exc
      self._client = AsyncCambAI(api_key=self._api_key, timeout=self._timeout)
    return self._client

  # ------------------------------------------------------------------
  # Polling
  # ------------------------------------------------------------------

  async def poll_async(
      self,
      get_status_fn: Any,
      task_id: Any,
      *,
      run_id: Any = None,
  ) -> Any:
    """Poll a camb.ai async task until it completes or fails."""
    for _ in range(self._max_poll_attempts):
      status = await get_status_fn(task_id, run_id=run_id)
      if hasattr(status, "status"):
        val = status.status
        if val in ("completed", "SUCCESS"):
          return status
        if val in ("failed", "FAILED", "error"):
          raise RuntimeError(
              f"Task failed: {getattr(status, 'error', 'Unknown error')}"
          )
      await asyncio.sleep(self._poll_interval)
    raise TimeoutError(
        f"Task {task_id} did not complete within "
        f"{self._max_poll_attempts * self._poll_interval}s"
    )

  # ------------------------------------------------------------------
  # Audio helpers
  # ------------------------------------------------------------------

  @staticmethod
  def detect_audio_format(data: bytes, content_type: str = "") -> str:
    """Detect audio format from raw bytes or content-type header."""
    if data.startswith(b"RIFF"):
      return "wav"
    if data.startswith((b"\xff\xfb", b"\xff\xfa", b"ID3")):
      return "mp3"
    if data.startswith(b"fLaC"):
      return "flac"
    if data.startswith(b"OggS"):
      return "ogg"
    ct = content_type.lower()
    for key, fmt in [
        ("wav", "wav"),
        ("wave", "wav"),
        ("mpeg", "mp3"),
        ("mp3", "mp3"),
        ("flac", "flac"),
        ("ogg", "ogg"),
    ]:
      if key in ct:
        return fmt
    return "pcm"

  @staticmethod
  def add_wav_header(pcm_data: bytes) -> bytes:
    """Wrap raw PCM data (24 kHz, mono, 16-bit) with a WAV header."""
    sr, ch, bps = 24000, 1, 16
    byte_rate = sr * ch * bps // 8
    block_align = ch * bps // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        ch,
        sr,
        byte_rate,
        block_align,
        bps,
        b"data",
        data_size,
    )
    return header + pcm_data

  @staticmethod
  def save_audio(data: bytes, suffix: str = ".wav") -> str:
    """Save audio bytes to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
      f.write(data)
      return f.name

  # ------------------------------------------------------------------
  # Formatting helpers
  # ------------------------------------------------------------------

  @staticmethod
  def gender_str(g: int) -> str:
    """Convert a numeric gender code to a human-readable string."""
    return {
        0: "not_specified",
        1: "male",
        2: "female",
        9: "not_applicable",
    }.get(g, "unknown")

  @staticmethod
  def format_transcription(transcription: Any) -> str:
    """Format a transcription result as a JSON string."""
    out: dict[str, Any] = {
        "text": getattr(transcription, "text", ""),
        "segments": [],
        "speakers": [],
    }
    if hasattr(transcription, "segments"):
      for seg in transcription.segments:
        out["segments"].append({
            "start": getattr(seg, "start", 0),
            "end": getattr(seg, "end", 0),
            "text": getattr(seg, "text", ""),
            "speaker": getattr(seg, "speaker", None),
        })
    if hasattr(transcription, "speakers"):
      out["speakers"] = list(transcription.speakers)
    elif out["segments"]:
      out["speakers"] = list(
          {s["speaker"] for s in out["segments"] if s.get("speaker")}
      )
    return json.dumps(out, indent=2)

  def format_voices(self, voices: Any) -> str:
    """Format a list of voice objects as a JSON string."""
    out: list[dict[str, Any]] = []
    for v in voices:
      if isinstance(v, dict):
        out.append({
            "id": v.get("id"),
            "name": v.get("voice_name", v.get("name", "Unknown")),
            "gender": self.gender_str(v.get("gender", 0)),
            "age": v.get("age"),
            "language": v.get("language"),
        })
      else:
        out.append({
            "id": getattr(v, "id", None),
            "name": getattr(v, "voice_name", getattr(v, "name", "Unknown")),
            "gender": self.gender_str(getattr(v, "gender", 0)),
            "age": getattr(v, "age", None),
            "language": getattr(v, "language", None),
        })
    return json.dumps(out, indent=2)

  def format_separation(self, sep: Any) -> str:
    """Format an audio-separation result as a JSON string."""
    out: dict[str, Any] = {
        "vocals": None,
        "background": None,
        "status": "completed",
    }
    for attr, key in [
        ("vocals_url", "vocals"),
        ("vocals", "vocals"),
        ("voice_url", "vocals"),
        ("background_url", "background"),
        ("background", "background"),
        ("instrumental_url", "background"),
    ]:
      val = getattr(sep, attr, None)
      if val and out[key] is None:
        if isinstance(val, bytes):
          out[key] = self.save_audio(val, f"_{key}.wav")
        else:
          out[key] = val
    return json.dumps(out, indent=2)

  @staticmethod
  def extract_translation(result: Any) -> str:
    """Extract translated text from an SDK result."""
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
      parts: list[str] = []
      for chunk in result:
        if hasattr(chunk, "text"):
          parts.append(chunk.text)
        elif isinstance(chunk, str):
          parts.append(chunk)
      return "".join(parts)
    if hasattr(result, "text"):
      return str(result.text)
    return str(result)
