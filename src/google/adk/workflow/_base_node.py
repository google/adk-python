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

from collections.abc import AsyncGenerator
from typing import Any
from typing import final
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import TypeAdapter
from pydantic import ValidationError

from ..utils._schema_utils import SchemaType
from ._retry_config import RetryConfig

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..events.event import Event


class BaseNode(BaseModel):
  """A base class for all nodes in the workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(...)
  """The unique name of the node within the workflow graph."""

  @field_validator('name')
  @classmethod
  def _validate_name(cls, v: str) -> str:
    if not v.isidentifier():
      raise ValueError(f"Node name '{v}' must be a valid Python identifier.")
    return v

  description: str = ''
  """A human-readable description of what this node does."""

  rerun_on_resume: bool = False
  """Controls behavior when resuming after an interrupt.

  If True, the node reruns from scratch. If False, it completes immediately
  using the user's resuming input as the node's output.
  """

  wait_for_output: bool = False
  """If True, node only transitions to COMPLETED upon yielding output or route.

  Without output/route, the node enters WAITING state and downstream nodes are
  not triggered, allowing predecessors to re-trigger it. This is useful for nodes
  like ``JoinNode`` that run multiple times before producing a final output.

  WARNING: Completing execution without ever yielding output/route causes an
  indefinite WAITING state (deadlock). This is considered a user configuration error.
  """

  retry_config: RetryConfig | None = None
  """Configuration for retrying the node on failure.

  If set, exceptions raised by the node will trigger retries according
  to the specified policy.
  """

  timeout: float | None = None
  """Maximum time in seconds for this node to complete.

  If the node does not finish within this duration, it is cancelled and
  treated as a failure (raising ``NodeTimeoutError``).  This integrates
  with ``retry_config`` — a timed-out node can be retried if retries
  are configured.

  ``None`` means no timeout (the node runs until completion).
  """

  input_schema: SchemaType | None = None
  """Schema to validate and coerce node input data.

  Supports all ``SchemaType`` variants. Validation uses ``TypeAdapter``
  and runs centrally in the node runner before ``node.run()`` is called.

  ``None`` means no input validation (the default).
  """

  output_schema: SchemaType | None = None
  """Schema to validate and coerce node output data.

  Supports all ``SchemaType`` variants (Pydantic ``BaseModel`` subclass,
  generic aliases like ``list[str]``, raw ``dict`` schemas, etc.).

  When set to a ``BaseModel`` subclass, the node's output data is validated:
    - dict → ``output_schema.model_validate(data).model_dump()``
    - BaseModel instance → ``data.model_dump()`` (already converted)

  ``None`` means no output validation (the default).
  """

  state_schema: type[BaseModel] | None = None
  """Optional Pydantic model declaring the expected state keys and types.

  When set, ``ctx.state`` mutations are validated at runtime against
  this schema.  Child nodes inherit the schema from their parent
  (via InvocationContext) unless they declare their own.

  Prefixed keys (``app:``, ``user:``, ``temp:``) bypass validation.
  """

  def _validate_schema(self, data: Any, schema: Any) -> Any:
    """Validates data against a schema using ``TypeAdapter``.

    Handles BaseModel, list[BaseModel], primitive types, unions, and
    generic aliases.  Descriptive schemas (``types.Schema``, raw
    ``dict``) are skipped.  Any BaseModel instances in the validated
    result are converted to dicts via ``model_dump()`` to keep
    ``Event.output`` JSON-serializable.
    """
    if data is None or schema is None:
      return data

    if isinstance(schema, (dict, types.Schema)):
      return data

    validated = TypeAdapter(schema).validate_python(data)
    return self._to_serializable(validated)

  def _validate_input_data(self, data: Any) -> Any:
    """Validates data against input_schema if set."""
    if self.input_schema and isinstance(data, types.Content):
      # Extract text from Content (e.g. user input from START node).
      text = ''.join(part.text for part in data.parts if part.text)
      if self.input_schema is str:
        return text
      # If schema is defined, try to parse the text as JSON.
      try:
        return TypeAdapter(self.input_schema).validate_json(text)
      except ValidationError:
        # Fallback to validate_python if it's a raw string matching the schema.
        try:
          return TypeAdapter(self.input_schema).validate_python(text)
        except ValidationError:
          pass
    return self._validate_schema(data, self.input_schema)

  def _validate_output_data(self, data: Any) -> Any:
    """Validates data against output_schema if set."""
    if not self.output_schema:
      return data

    # 1. Try standard validation first
    try:
      return self._validate_schema(data, self.output_schema)
    except ValidationError as e:
      # 2. If failed, try to parse JSON ONLY if it's Content
      if isinstance(data, types.Content):
        text = ''.join(part.text for part in data.parts if part.text)
        if self.output_schema is str:
          return text
        if text.strip():
          try:
            validated = TypeAdapter(self.output_schema).validate_json(text)
            return self._to_serializable(validated)
          except ValidationError:
            pass

      # 3. If not Content or parsing failed, re-raise original error
      raise e

  @staticmethod
  def _to_serializable(data: Any) -> Any:
    """Converts BaseModel instances to dicts recursively."""
    if isinstance(data, BaseModel):
      return data.model_dump()
    if isinstance(data, list):
      return [BaseNode._to_serializable(item) for item in data]
    if isinstance(data, dict):
      return {k: BaseNode._to_serializable(v) for k, v in data.items()}
    return data

  @final
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Event, None]:
    """Public entry point. Calls _run_impl, normalizes yields to Event.

    Normalization rules:
    - None -> skipped
    - Event -> pass through
    - RequestInput -> convert to interrupt Event
    - Any other value -> Event(output=value)
    """
    from ..events.event import Event
    from ..events.request_input import RequestInput
    from ..utils.context_utils import Aclosing

    node_input = self._validate_input_data(node_input)
    async with Aclosing(self._run_impl(ctx=ctx, node_input=node_input)) as agen:
      async for item in agen:
        if item is None:
          continue
        if isinstance(item, Event):
          if item.output is not None:
            item.output = self._validate_output_data(item.output)
          yield item
        elif isinstance(item, RequestInput):
          from .utils._workflow_hitl_utils import create_request_input_event

          yield create_request_input_event(item)
        else:
          validated = self._validate_output_data(item)
          yield Event(output=validated)

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Override point for node execution logic.

    Yields any of: Event, RequestInput, raw data, or None.
    BaseNode.run() normalizes all yields to Event before the caller
    sees them.
    """
    raise NotImplementedError(
        f'_run_impl for {type(self).__name__} is not implemented.'
    )
    yield  # AsyncGenerator requires at least one yield statement

  @property
  def _requires_all_predecessors(self) -> bool:
    """If True, the node waits for all predecessors to complete before running."""
    return False


START = BaseNode(name='__START__')
"""Sentinel node marking the entry point of a workflow graph.

START is never executed — ``Workflow._seed_start_triggers`` bypasses it
and seeds triggers for its successors directly.
"""
