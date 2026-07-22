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

"""Base model for serialized Pydantic models."""

from __future__ import annotations

import pydantic
from pydantic import alias_generators


class SerializedBaseModel(pydantic.BaseModel):
  """Base model for all Pydantic models that are serialized for Web Server or Storage.

  This model enforces camelCase serialization by default to align with JSON
  conventions used in the web UI and external APIs, while allowing Python code
  to use snake_case.

  Note: `model_dump_json()` is overridden to use `by_alias=True` by default to
  ensure camelCase output in JSON serialization.
  """

  model_config = pydantic.ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      use_attribute_docstrings=True,
  )

  def model_dump_json(self, **kwargs) -> str:
    """Override model_dump_json to use by_alias=True by default."""
    kwargs.setdefault('by_alias', True)
    return super().model_dump_json(**kwargs)
