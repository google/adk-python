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

from typing import Any
from typing import Optional

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental


@experimental
class ToolConfirmation(BaseModel):
  """Represents a tool confirmation configuration."""

  model_config = ConfigDict(
      extra="forbid",
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  hint: str = ""
  """The hint text for why the input is needed."""
  confirmed: bool = False
  """Whether the tool excution is confirmed."""
  payload: Optional[Any] = None
  """The custom data payload needed from the user to continue the flow.
  It should be JSON serializable."""
