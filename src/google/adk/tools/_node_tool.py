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
from typing_extensions import override

from ..utils._schema_utils import schema_to_json_schema
from ..workflow._base_node import BaseNode
from ..workflow._errors import NodeInterruptedError
from .base_tool import BaseTool
from .tool_context import ToolContext


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
      orig_input_schema = getattr(node, 'input_schema', None)
      orig_output_schema = getattr(node, 'output_schema', None)
      node = FunctionNode(
          func=node._func,
          name=node.name,
          rerun_on_resume=node.rerun_on_resume,
          retry_config=node.retry_config,
          timeout=node.timeout,
          auth_config=node.auth_config,
          parameter_binding='node_input',  # Force binding to node_input
          state_schema=node.state_schema,
      )
      if orig_input_schema is not None:
        node.input_schema = orig_input_schema
      if orig_output_schema is not None:
        node.output_schema = orig_output_schema

    if not getattr(node, 'input_schema', None):
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
    self.is_long_running = True

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    schema = schema_to_json_schema(self.node.input_schema)

    # The GenAI API strictly requires parameters_json_schema to be an 'object'
    # type schema. If the node has a primitive input schema (e.g., str, int),
    # we wrap it into an object schema with a 'request' property.
    if isinstance(schema, dict) and schema.get('type') != 'object':
      schema = {
          'type': 'object',
          'properties': {
              'request': schema,
          },
          'required': ['request'],
      }

    decl = types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema=schema,
    )

    output_schema = getattr(self.node, 'output_schema', None)
    if output_schema:
      decl.response_json_schema = schema_to_json_schema(output_schema)

    return decl

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    import inspect

    from pydantic import BaseModel

    input_schema = self.node.input_schema
    node_input: Any
    if inspect.isclass(input_schema) and issubclass(input_schema, BaseModel):
      try:
        # Convert input based on Pydantic schema
        node_input = input_schema.model_validate(args)
      except Exception as e:
        return f'Error validating input for node: {e}'
    else:
      schema = schema_to_json_schema(input_schema)
      if isinstance(schema, dict) and schema.get('type') != 'object':
        node_input = args.get('request')
      else:
        node_input = args

    fc_id = tool_context.function_call_id
    base_branch = tool_context.branch
    segment = f'{self.name}@{fc_id}'
    tool_branch = f'{base_branch}.{segment}' if base_branch else segment

    try:
      return await tool_context.run_node(
          self.node,
          node_input=node_input,
          override_branch=tool_branch,
          use_sub_branch=False,
          raise_on_wait=True,
      )
    except NodeInterruptedError as nie:
      # Propagates the interrupt up so the runner pauses the invocation
      raise nie
    except Exception as e:
      return f'Error running node {self.name}: {e}'
