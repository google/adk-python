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

"""Tests for _audio_utils.

Verifies that the audio helpers parse sample rates from MIME types and
resample 16-bit PCM to the Live API input rate.
"""

from __future__ import annotations

import array
import logging

from google.adk.evaluation import _audio_utils as audio_utils
import pytest


def _pcm(samples: list[int]) -> bytes:
  """Builds little-endian signed 16-bit PCM bytes from integer samples."""
  return array.array("h", samples).tobytes()


def _samples(pcm: bytes) -> list[int]:
  """Decodes little-endian signed 16-bit PCM bytes back into samples."""
  decoded = array.array("h")
  decoded.frombytes(pcm)
  return decoded.tolist()


# ---------------------------------------------------------------------------
# parse_sample_rate
# ---------------------------------------------------------------------------


def test_parse_sample_rate_extracts_rate_parameter():
  """A mime type carrying a rate parameter yields that rate."""
  assert audio_utils.parse_sample_rate("audio/l16; rate=24000", 8000) == 24000


def test_parse_sample_rate_without_rate_returns_default():
  """A mime type without a rate parameter falls back to the default."""
  assert audio_utils.parse_sample_rate("audio/pcm", 16000) == 16000


def test_parse_sample_rate_none_returns_default():
  """A missing mime type falls back to the default."""
  assert audio_utils.parse_sample_rate(None, 16000) == 16000


def test_parse_sample_rate_ignores_rate_substrings():
  """A parameter containing rate as a substring is not a sample rate."""
  assert (
      audio_utils.parse_sample_rate("audio/pcm;bitrate=128000", 24000) == 24000
  )


def test_parse_sample_rate_is_case_insensitive():
  """The rate parameter name is parsed case-insensitively."""
  assert audio_utils.parse_sample_rate("audio/pcm;RATE=16000", 24000) == 16000


# ---------------------------------------------------------------------------
# resample_pcm16
# ---------------------------------------------------------------------------


def test_resample_matching_rates_returns_input_unchanged():
  """Resampling with equal source and target rates is a no-op."""
  pcm = _pcm([1, 2, 3, 4])

  assert audio_utils.resample_pcm16(pcm, 16000, 16000) == pcm


def test_resample_empty_input_returns_empty():
  """Resampling empty audio yields empty audio."""
  assert audio_utils.resample_pcm16(b"", 24000, 16000) == b""


def test_resample_zero_source_rate_raises():
  """A zero source sample rate is rejected before resampling."""
  with pytest.raises(ValueError, match="Sample rates must be positive"):
    audio_utils.resample_pcm16(_pcm([1, 2]), 0, 16000)


def test_resample_zero_target_rate_raises():
  """A zero target sample rate is rejected before resampling."""
  with pytest.raises(ValueError, match="Sample rates must be positive"):
    audio_utils.resample_pcm16(_pcm([1, 2]), 24000, 0)


def test_resample_single_sample_returns_input_unchanged():
  """Audio too short to interpolate is returned unchanged."""
  pcm = _pcm([42])

  assert audio_utils.resample_pcm16(pcm, 24000, 16000) == pcm


def test_resample_downsamples_by_rate_ratio():
  """Downsampling 24 kHz to 16 kHz scales the sample count by 2/3."""
  pcm = _pcm(list(range(600)))

  result = audio_utils.resample_pcm16(pcm, 24000, 16000)

  assert len(_samples(result)) == 400


def test_resample_interpolates_between_samples():
  """A downsampled point is the linear interpolation of its neighbors."""
  # Source samples 0..3 at 24 kHz; target index 1 maps to src_pos 1.5,
  # i.e. halfway between samples[1]=100 and samples[2]=200 -> 150.
  pcm = _pcm([0, 100, 200, 300])

  result = _samples(audio_utils.resample_pcm16(pcm, 24000, 16000))

  assert result[1] == 150


# ---------------------------------------------------------------------------
# to_live_input
# ---------------------------------------------------------------------------


def test_to_live_input_resamples_from_declared_rate():
  """Audio tagged at 24 kHz is resampled to the Live input sample count."""
  pcm = _pcm(list(range(600)))

  result = audio_utils.to_live_input(pcm, "audio/l16; rate=24000")

  assert len(_samples(result)) == 400


def test_to_live_input_defaults_to_common_tts_rate_and_warns(caplog):
  """Audio with no declared rate defaults to the common TTS rate and warns."""
  pcm = _pcm(list(range(600)))

  with caplog.at_level(logging.WARNING, logger=audio_utils.logger.name):
    result = audio_utils.to_live_input(pcm, "audio/pcm")

  # 24 kHz default downsamples 600 samples to 16 kHz (400 samples)...
  assert len(_samples(result)) == 400
  # ...and the unparseable rate warns rather than silently guessing.
  assert any(
      "no `rate=`" in record.message and record.levelno == logging.WARNING
      for record in caplog.records
  )


def test_to_live_input_does_not_warn_when_rate_is_declared(caplog):
  """A declared source rate resamples without emitting a warning."""
  pcm = _pcm(list(range(600)))

  with caplog.at_level(logging.WARNING, logger=audio_utils.logger.name):
    audio_utils.to_live_input(pcm, "audio/l16; rate=24000")

  assert not caplog.records


def test_to_live_input_at_target_rate_is_unchanged():
  """Audio already at the Live input rate passes through unchanged."""
  pcm = _pcm([1, 2, 3, 4])

  result = audio_utils.to_live_input(pcm, "audio/pcm;rate=16000")

  assert result == pcm
