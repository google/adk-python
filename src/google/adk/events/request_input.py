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
from __future__ import annotations

from typing import Any
from typing import Optional
import uuid

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils._schema_utils import SchemaType


class RequestInput(BaseModel):
  """Represents a request for input from the user."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  interrupt_id: str = Field(
      description=(
          "The ID of the interrupt, usually a function call ID. This is used"
          " to identify the interrupt that the input is for."
      ),
      default_factory=lambda: str(uuid.uuid4()),
  )
  """The ID of the interrupt, usually a function call ID.

  Reusing the same interrupt_id across loop iterations (e.g. a
  rejection/retry cycle) is supported — the framework matches
  function calls and responses by count. Using unique IDs per
  iteration is still recommended for clarity in event logs.
  """
  payload: Optional[Any] = None
  """ Custom payload to be provided for resuming."""
  message: Optional[str] = Field(
      None,
      description="A message to display to the user when requesting input.",
  )
  """A message to display to the user when requesting input."""
  response_schema: Optional[SchemaType] = Field(
      None,
      description=(
          "The expected schema of the response. Accepts a Python type"
          " (e.g. a Pydantic BaseModel class), a generic alias"
          " (e.g. list[str]), or a raw JSON Schema dict."
          " If None, it defaults to Any."
      ),
  )
  """The expected schema of the response."""
