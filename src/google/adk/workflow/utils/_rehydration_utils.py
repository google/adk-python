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

"""Utilities for ADK workflow rehydration."""

from dataclasses import dataclass
from dataclasses import field
import json
from typing import Any

from google.genai import types
from pydantic import TypeAdapter
from pydantic import ValidationError

from ...events.event import Event
from ...events.request_input import RequestInput
from ...events._node_path_builder import _NodePathBuilder
from ._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME

_RESULT_KEY = 'result'


@dataclass
class _ChildScanState:
  """State accumulated for a child node during event scanning."""

  run_id: str | None = None
  output: Any = None
  route: str | None = None
  branch: str | None = None
  interrupt_ids: set[str] = field(default_factory=set)
  resolved_ids: set[str] = field(default_factory=set)
  resolved_responses: dict[str, Any] = field(default_factory=dict)


def _wrap_response(value: Any) -> dict[str, Any]:
  """Wraps a value into a dict suitable for FunctionResponse.response.

  If the value is already a dict, returns it as-is.
  Otherwise wraps as ``{"result": value}``.
  """
  if isinstance(value, dict):
    return value
  return {_RESULT_KEY: value}


def _unwrap_response(data: Any) -> Any:
  """Unwraps a FunctionResponse dict to the original value.

  If ``data`` is a dict with exactly one key ``"result"``, extracts the
  value.  String values are JSON-parsed when possible (the web frontend
  wraps user text as ``{"result": text}`` without parsing).

  Otherwise returns ``data`` unchanged.
  """
  if isinstance(data, dict) and len(data) == 1 and _RESULT_KEY in data:
    value = data[_RESULT_KEY]
    if isinstance(value, str):
      try:
        value = json.loads(value)
      except (json.JSONDecodeError, ValueError):
        pass
      return value
    return value
  return data


def _extract_schema_from_event(event: Event, interrupt_id: str) -> Any | None:
  """Extracts the response schema from an event if it's a RequestInput call."""
  if not event.content or not event.content.parts:
    return None

  for part in event.content.parts:
    fc = part.function_call
    if (
        fc
        and fc.name == REQUEST_INPUT_FUNCTION_CALL_NAME
        and fc.id == interrupt_id
    ):
      return fc.args.get('response_schema')

  return None


def _validate_resume_response(response_data: Any, schema: Any) -> Any:
  """Validates and coerces resume response data against a schema.

  Args:
    response_data: The data to validate.
    schema: The schema to validate against (Python type, GenericAlias, or raw
      JSON Schema dict).

  Returns:
    The validated and coerced data.
  """
  if schema is None:
    return response_data

  # If it's a JSON Schema dict, map type to Python type for TypeAdapter
  if isinstance(schema, dict):
    type_str = schema.get('type')

    type_mapping = {
        'integer': int,
        'number': float,
        'string': str,
        'boolean': bool,
        'array': list,
        'object': dict,
    }

    # Special handling for object schemas with properties
    if type_str == 'object' and 'properties' in schema:
      from pydantic import create_model

      properties = schema['properties']
      required = schema.get('required', [])

      fields = {}
      for prop_name, prop_schema in properties.items():
        prop_type_str = prop_schema.get('type')
        prop_type = (
            type_mapping.get(prop_type_str, Any) if prop_type_str else Any
        )

        if prop_name in required:
          fields[prop_name] = (prop_type, ...)
        else:
          fields[prop_name] = (prop_type | None, None)  # type: ignore[assignment]

      try:
        DynamicModel = create_model('DynamicModel', **fields)
        # Validate and return as dict
        model_instance = TypeAdapter(DynamicModel).validate_python(
            response_data
        )
        return model_instance.model_dump()
      except ValidationError as e:
        raise ValueError(f'Validation failed for object schema: {e}') from e

    mapped_type = type_mapping.get(type_str) if type_str else None
    if mapped_type:
      try:
        return TypeAdapter(mapped_type).validate_python(response_data)
      except ValidationError as e:
        raise ValueError(f'Failed to coerce data to {type_str}: {e}') from e

    # Fallback: skip validation for complex schemas (similar to base node)
    return response_data

  # For Python types and Pydantic models, use TypeAdapter directly
  try:
    return TypeAdapter(schema).validate_python(response_data)
  except ValidationError as e:
    raise ValueError(f'Validation failed against schema: {e}') from e


def _process_content_object(event: Event) -> Any:
  """Extracts output from event.content."""
  if not event.content or not event.content.parts:
    return None

  text = ''.join(
      p.text for p in event.content.parts if p.text and not p.thought
  )
  if not text:
    return None

  text = text.strip()
  try:
    return json.loads(text)
  except (json.JSONDecodeError, ValueError):
    return text


def _reconstruct_node_states(
    events: list[Event],
    base_path: str,
    invocation_id: str,
    group_by_direct_child: bool = False,
) -> dict[str, _ChildScanState]:
  """Scans session events to reconstruct node states for resume."""
  scan_states: dict[str, _ChildScanState] = {}
  interrupt_owner: dict[str, str] = {}
  schemas_by_id: dict[str, Any] = {}

  base_path_builder = _NodePathBuilder.from_string(base_path)

  def get_owner_key(event_path_builder: _NodePathBuilder) -> str | None:
    if group_by_direct_child:
      if not event_path_builder.is_descendant_of(base_path_builder):
        return None
      child_path = base_path_builder.get_direct_child(event_path_builder)
      segment: str = child_path._segments[-1]
      return segment
    else:
      if event_path_builder == base_path_builder or event_path_builder.is_descendant_of(
          base_path_builder
      ):
        return base_path
      return None

  for event in events:
    if invocation_id and event.invocation_id != invocation_id:
      continue

    # 1. Handle FunctionResponse (User responses to interrupts)
    if event.author == 'user' and event.content and event.content.parts:
      for part in event.content.parts:
        fr = part.function_response
        if fr and fr.id and fr.id in interrupt_owner:
          owner = interrupt_owner[fr.id]
          if owner not in scan_states:
            scan_states[owner] = _ChildScanState()
          scan_states[owner].resolved_ids.add(fr.id)
          response_data = _unwrap_response(fr.response)

          schema = schemas_by_id.get(fr.id)
          if schema:
            try:
              response_data = _validate_resume_response(response_data, schema)
            except ValueError as e:
              raise ValueError(
                  f'Validation failed for interrupt {fr.id}: {e}'
              ) from e

          scan_states[owner].resolved_responses[fr.id] = response_data
      continue

    # 2. Match events under base_path
    event_node_path = event.node_info.path or ''
    event_path_builder = _NodePathBuilder.from_string(event_node_path)
    owner_key = get_owner_key(event_path_builder)

    if not owner_key:
      continue

    # 3. Initialize state for the owner if needed
    if owner_key not in scan_states:
      owner_path_builder = _NodePathBuilder.from_string(owner_key)
      scan_states[owner_key] = _ChildScanState(run_id=owner_path_builder.run_id)

    child = scan_states[owner_key]

    # 4. Determine if event is direct child or delegated output
    is_direct = False
    if group_by_direct_child:
      is_direct = event_path_builder.is_direct_child_of(base_path_builder)
    else:
      is_direct = event_path_builder == base_path_builder

    has_output = event.output is not None
    use_message_as_output = False
    if (
        not has_output
        and event.node_info
        and event.node_info.message_as_output
        and event.content is not None
    ):
      has_output = True
      use_message_as_output = True

    is_delegated = False
    if has_output and event.node_info.output_for:
      if not group_by_direct_child:
        is_delegated = base_path in event.node_info.output_for

    # 5. Extract output and route
    if is_direct or is_delegated:
      if event.output is not None:
        child.output = event.output
        child.branch = event.branch
      elif use_message_as_output:
        child.output = _process_content_object(event)
      if event.actions and event.actions.route is not None:
        child.route = event.actions.route

    # 6. Extract interrupts and their schemas
    if event.long_running_tool_ids:
      for interrupt_id in event.long_running_tool_ids:
        child.interrupt_ids.add(interrupt_id)
        interrupt_owner[interrupt_id] = owner_key

        schema_json = _extract_schema_from_event(event, interrupt_id)
        if schema_json:
          schemas_by_id[interrupt_id] = schema_json

  return scan_states
