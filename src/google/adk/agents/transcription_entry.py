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

from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict


class TranscriptionEntry(BaseModel):
  """Store the data that can be used for transcription."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  role: Optional[str] = None
  """The role that created this data, typically "user" or "model". For function 
  call, this is None."""

  data: Union[types.Blob, types.Content]
  """The data that can be used for transcription"""
