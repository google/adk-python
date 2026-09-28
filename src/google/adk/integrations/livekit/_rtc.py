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

"""The one place the LiveKit media SDK is imported."""

from __future__ import annotations

try:
  import livekit.rtc as rtc
except ImportError as e:
  raise ImportError(
      "livekit is not installed. Please install it with "
      '`pip install "google-adk[livekit]"`.'
  ) from e

# `Room.on` takes a Literal of event names, not a str; rtc hides this one.
EventTypes = rtc.room.EventTypes

__all__ = ["EventTypes", "rtc"]
