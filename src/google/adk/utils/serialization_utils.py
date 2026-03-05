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

"""Shared serialization utilities for Pydantic models."""

import json
from typing import Any
from typing import Optional

from pydantic import BaseModel


def serialize_pydantic_model(obj: Any) -> Optional[str]:
  """Serialize a Pydantic BaseModel to a JSON string.

  Args:
    obj: The object to check and serialize.

  Returns:
    A JSON string if the object is a Pydantic BaseModel, or None otherwise.
  """
  if isinstance(obj, BaseModel):
    return json.dumps(obj.model_dump(), default=str)
  return None
