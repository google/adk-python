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

from google.genai import types
from pydantic import ValidationError
from typing_extensions import override

from ..utils._schema_utils import schema_to_json_schema
from ..workflow._base_node import BaseNode
from ..workflow._errors import DynamicNodeFailError
from ..workflow._errors import WorkflowDataError
from .base_tool import BaseTool
from .tool_context import ToolContext


def _build_node_declaration(
    node: BaseNode,
    *,
    name: str | None = None,
    description: str | None = None,
) -> types.FunctionDeclaration:
  """Builds a FunctionDeclaration exposing a BaseNode as a callable tool."""
  from ..workflow._function_node import FunctionNode

  if (
      isinstance(node, FunctionNode)
      and node.parameter_binding != 'node_input'
      and node.input_schema is None
  ):
    node = node._as_tool_node()

  decl = types.FunctionDeclaration(
      name=name or node.name,
      description=description
      or node.description
      or f'Executes the node: {node.name}',
  )

  input_schema = getattr(node, 'input_schema', None)
  if input_schema is not None:
    schema = schema_to_json_schema(input_schema)
    # The GenAI API strictly requires parameters_json_schema to be an 'object'
    # type schema. If the node has a primitive input schema (e.g., str, int),
    # wrap it into an object schema with a 'request' property.
    if isinstance(schema, dict) and schema.get('type') != 'object':
      schema = {
          'type': 'object',
          'properties': {
              'request': schema,
          },
          'required': ['request'],
      }
    decl.parameters_json_schema = schema

  output_schema = getattr(node, 'output_schema', None)
  if output_schema is not None:
    decl.response_json_schema = schema_to_json_schema(output_schema)

  return decl


class NodeTool(BaseTool):
  """A tool wrapper that executes a BaseNode (e.g. a Workflow or loop node)."""

  def __init__(
      self,
      node: BaseNode,
      name: str | None = None,
      description: str | None = None,
  ):
    from ..agents.base_agent import BaseAgent
    from ..workflow._function_node import FunctionNode

    if isinstance(node, BaseAgent):
      raise ValueError(
          f"Agent '{node.name}' cannot be wrapped as a NodeTool. Agents should"
          ' be invoked as Sub-Agents instead.'
      )

    # Automatically align FunctionNode binding
    if (
        isinstance(node, FunctionNode)
        and node.parameter_binding != 'node_input'
    ):
      node = node._as_tool_node()

    # A FunctionNode has already inferred its schema by here, and that yields
    # None only when the function has nothing to bind.
    if not isinstance(node, FunctionNode) and not getattr(
        node, 'input_schema', None
    ):
      raise ValueError(
          f"Node '{node.name}' does not have an input_schema defined."
          ' NodeTool requires an explicit Pydantic input_schema on the wrapped'
          ' node.'
      )

    self.node = node
    super().__init__(
        name=name or node.name,
        description=description
        or node.description
        or f'Executes the node: {node.name}',
    )

  @override
  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return _build_node_declaration(
        self.node,
        name=self.name,
        description=self.description,
    )

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    input_schema = getattr(self.node, 'input_schema', None)
    schema = (
        schema_to_json_schema(input_schema)
        if input_schema is not None
        else None
    )
    if isinstance(schema, dict) and schema.get('type') != 'object':
      node_input = args.get('request')
    else:
      node_input = args

    try:
      node_input = self.node._validate_input_data(node_input)
    except (ValidationError, WorkflowDataError) as e:
      # Same shape as FunctionTool's argument validation errors, so the
      # model can correct its arguments and retry.
      return {
          'error': (
              f'Invoking `{self.name}()` failed due to argument validation'
              f' errors:\n{e}\nYou could retry calling this tool with'
              ' corrected argument types.'
          )
      }

    res = await _run_node_in_tool_context(
        self.node,
        tool_name=self.name,
        node_input=node_input,
        tool_context=tool_context,
    )
    if res is None:
      return {'result': None}
    return res


async def _run_node_in_tool_context(
    node: BaseNode,
    *,
    tool_name: str,
    node_input: Any,
    tool_context: ToolContext,
    key_run_by_function_call: bool = False,
) -> Any:
  """Executes a BaseNode within a ToolContext on an isolated tool branch.

  Args:
    node: The node to execute.
    tool_name: The tool name, used as the branch segment.
    node_input: The input passed to the node.
    tool_context: The calling tool's context.
    key_run_by_function_call: Whether to use the function call id as the
      child's run_id, so repeated calls of the same tool get distinct node
      paths and do not share resume state. Otherwise the scheduler assigns the
      run_id.
  """
  fc_id = tool_context.function_call_id
  base_branch = tool_context.branch
  segment = f'{tool_name}@{fc_id}' if fc_id else tool_name
  tool_branch = f'{base_branch}.{segment}' if base_branch else segment
  run_id = None
  if key_run_by_function_call and (
      tool_context._workflow_scheduler is None
      or (fc_id and not fc_id.isdigit())
  ):
    # Under a workflow scheduler, numeric ids are reserved for auto-generated
    # run_ids, so a numeric fc_id falls back to auto-generation.
    run_id = fc_id

  try:
    return await tool_context.run_node(
        node,
        node_input=node_input,
        run_id=run_id,
        override_branch=tool_branch,
        use_sub_branch=False,
        raise_on_wait=True,
    )
  except DynamicNodeFailError as e:
    # Surface the node's own error, as a FunctionTool would, so the tool
    # pipeline runs on_tool_error callbacks with the real cause.
    raise e.error from e
