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

"""FinishTaskTool: signals task completion and sets finish_task action."""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import TypeAdapter
from pydantic import ValidationError
from typing_extensions import override

from ....tools.base_tool import BaseTool
from ....utils._schema_utils import SchemaType
from ._task_models import _DefaultTaskOutput

if TYPE_CHECKING:
  from ....models.llm_request import LlmRequest
  from ....tools.tool_context import ToolContext
  from ...llm_agent import LlmAgent

# Name of the finish_task tool
FINISH_TASK_TOOL_NAME = 'finish_task'

# Success result returned by FinishTaskTool.run_async when validation
# passes.  The wrapper uses this to distinguish a successful completion
# from a validation-error retry signal.
FINISH_TASK_SUCCESS_RESULT = 'Task completed.'


class FinishTaskTool(BaseTool):
  """Tool for signaling LlmAgent task completion.

  This tool allows the model to signal that the agent has completed its
  task. On success it sets ``tool_context.actions.finish_task`` with a
  serialized ``TaskResult`` dict.
  """

  def __init__(
      self,
      task_agent: LlmAgent,
  ):
    """Initialize the finish_task tool.

    Args:
      task_agent: The task agent this tool belongs to. The agent's
        ``output_schema`` is used for validation. If None, the default
        schema (a single ``result`` string) is used.
    """
    self._task_agent_name = task_agent.name

    output_schema = task_agent.output_schema
    self.output_schema: SchemaType = (
        output_schema if output_schema is not None else _DefaultTaskOutput
    )
    self._adapter: TypeAdapter[Any] = TypeAdapter(self.output_schema)
    raw_schema = self._adapter.json_schema()
    # FunctionDeclaration parameters must be a JSON object schema.
    # If the schema is already an object (e.g. BaseModel), use it directly.
    # Otherwise wrap it in an object with a single key.
    self._wrapper_key: str | None = (
        None if raw_schema.get('type') == 'object' else 'result'
    )

    description = (
        'Signal that this agent has completed its delegated task. Call this'
        ' when you have finished your delegated task.'
    )
    if output_schema:
      description += ' Pass the required output data in the parameters.'

    super().__init__(
        name=FINISH_TASK_TOOL_NAME,
        description=description,
    )

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Get the function declaration for this tool."""
    raw_schema = self._adapter.json_schema()
    if self._wrapper_key:
      # Extract $defs to the root level so $ref pointers remain valid
      # after wrapping the schema inside an object property.
      defs = raw_schema.pop('$defs', None)
      schema_json = {
          'type': 'object',
          'properties': {self._wrapper_key: raw_schema},
          'required': [self._wrapper_key],
      }
      if defs:
        schema_json['$defs'] = defs
    else:
      schema_json = raw_schema

    return types.FunctionDeclaration(
        name=FINISH_TASK_TOOL_NAME,
        description=self.description,
        parameters_json_schema=schema_json,
    )

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Process the outgoing LLM request to add tool and instructions.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request.
    """
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    instruction = self._build_instruction()
    llm_request.append_instructions([instruction])

  def _build_instruction(self) -> str:
    """Build the finish_task instruction.

    Returns:
      Instruction text for the LLM about when to call finish_task.
    """
    return """\
Do NOT call `finish_task` prematurely. Use your available tools to
fully complete every aspect of the delegated task first. If the
task is unclear, ask the user for clarification before proceeding.
Once the task is fully complete, call `finish_task` by itself with
no accompanying text output."""

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> str | dict[str, str]:
    """Execute the finish_task tool.

    Validates args against the output schema and sets
    ``tool_context.actions.finish_task`` on success.

    Args:
      args: The arguments passed to the tool.
      tool_context: The tool execution context.

    Returns:
      Confirmation message, or error dict if validation fails.
    """
    try:
      raw_value = args.get(self._wrapper_key) if self._wrapper_key else args
      validated = self._adapter.validate_python(raw_value)
      validated_output = self._adapter.dump_python(validated, mode='json')
    except ValidationError as e:
      return {
          'error': (
              f'Invoking `{self.name}()` failed due to validation'
              f' errors:\n{e}\nYou could retry calling this tool, but'
              ' it is IMPORTANT for you to provide all the mandatory'
              ' parameters with correct types.'
          )
      }

    # do not write actions.finish_task. The LlmAgent
    # wrapper sniffs the finish_task FC's `output` arg directly to
    # set event.output on the task agent's run.
    del validated_output

    return FINISH_TASK_SUCCESS_RESULT
