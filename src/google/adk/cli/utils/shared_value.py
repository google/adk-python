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

from typing import Generic
from typing import TypeVar

import pydantic

T = TypeVar("T")


class SharedValue(pydantic.BaseModel, Generic[T]):
  """Simple wrapper around a value to allow modifying it from callbacks."""

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True,
  )
  value: T
