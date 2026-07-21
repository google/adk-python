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

"""Audio helpers for feeding synthesized speech into a Live API session.

TTS backends commonly emit 24 kHz PCM, but the Live API only accepts 16 kHz
PCM input, so simulated user audio must be resampled before it is sent.
"""

from __future__ import annotations

import array
import logging
import re

logger = logging.getLogger("google_adk." + __name__)

# Live API input/output sample rates (16-bit mono PCM).
LIVE_INPUT_RATE_HZ = 16000
LIVE_OUTPUT_RATE_HZ = 24000
LIVE_INPUT_MIME_TYPE = "audio/pcm;rate=16000"

_RATE_RE = re.compile(r"rate=(\d+)")


def parse_sample_rate(mime_type: str | None, default: int) -> int:
  """Extracts the sample rate from a mime type like 'audio/pcm;rate=24000'."""
  if not mime_type:
    return default
  match = _RATE_RE.search(mime_type)
  return int(match.group(1)) if match else default


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
  """Resamples 16-bit mono PCM via linear interpolation.

  Returns the input unchanged when the rates match or it is too short to
  interpolate, avoiding a heavy DSP dependency for speech relayed to a
  transcribing model.
  """
  if not pcm or src_rate == dst_rate:
    return pcm

  samples = array.array("h")
  # Drop a trailing odd byte so the buffer is a whole number of samples.
  samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
  if len(samples) < 2:
    return pcm

  ratio = src_rate / dst_rate
  out_len = max(1, int(len(samples) / ratio))
  out = array.array("h", bytes(2 * out_len))
  last_index = len(samples) - 1
  for i in range(out_len):
    src_pos = i * ratio
    left = int(src_pos)
    right = min(left + 1, last_index)
    frac = src_pos - left
    out[i] = int(samples[left] * (1.0 - frac) + samples[right] * frac)
  return out.tobytes()


def to_live_input(pcm: bytes, source_mime_type: str | None) -> bytes:
  """Resamples synthesized speech audio to 16 kHz PCM for Live API input."""
  # Warn instead of silently guessing: a wrong assumed rate mis-pitches the
  # resample and would otherwise pass unnoticed.
  if not source_mime_type or not _RATE_RE.search(source_mime_type):
    logger.warning(
        "Audio mime type %r has no `rate=`; assuming %d Hz before resampling"
        " to the Live API input rate. Mislabeled audio will be resampled"
        " incorrectly.",
        source_mime_type,
        LIVE_OUTPUT_RATE_HZ,
    )
  src_rate = parse_sample_rate(source_mime_type, default=LIVE_OUTPUT_RATE_HZ)
  return resample_pcm16(pcm, src_rate=src_rate, dst_rate=LIVE_INPUT_RATE_HZ)
