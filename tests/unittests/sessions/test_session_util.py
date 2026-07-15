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

from google.adk.sessions._session_util import decode_model
from google.genai import types


def test_decode_model_returns_none_for_none():
  assert decode_model(None, types.Content) is None


def test_decode_model_validates_dict():
  result = decode_model(
      {"role": "user", "parts": [{"text": "hello"}]}, types.Content
  )
  assert isinstance(result, types.Content)
  assert result.role == "user"
  assert result.parts[0].text == "hello"


def test_decode_model_returns_none_for_non_dict_value():
  # A transcription field persisted as the JSON string "null" instead of SQL
  # NULL should decode to None rather than crash session replay.
  assert decode_model("null", types.Transcription) is None
