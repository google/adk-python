# Copyright 2025 Google LLC
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

from __future__ import annotations

from google.genai import types


def is_audio_part(part: types.Part) -> bool:
  return (
      part.inline_data is not None
      and part.inline_data.mime_type is not None
      and part.inline_data.mime_type.startswith('audio/')
  ) or (
      part.file_data is not None
      and part.file_data.mime_type is not None
      and part.file_data.mime_type.startswith('audio/')
  )


def filter_audio_parts(content: types.Content) -> types.Content | None:
  if not content.parts:
    return None
  filtered_parts = [part for part in content.parts if not is_audio_part(part)]
  if not filtered_parts:
    return None
  return types.Content(role=content.role, parts=filtered_parts)
