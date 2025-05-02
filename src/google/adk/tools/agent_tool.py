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
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import model_validator
from typing_extensions import override

from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..runners import Runner
from ..sessions.in_memory_session_service import InMemorySessionService
from . import _automatic_function_calling_util
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.llm_agent import LlmAgent


class AgentTool(BaseTool):
  """A tool that wraps an agent.

  This tool allows an agent to be called as a tool within a larger application.
  The agent's input schema is used to define the tool's input parameters, and
  the agent's output is returned as the tool's result.

  Attributes:
    agent: The agent to wrap.
    skip_summarization: Whether to skip summarization of the agent output.
  """

  def __init__(self, agent: BaseAgent, skip_summarization: bool = False):
    self.agent = agent
    self.skip_summarization: bool = skip_summarization

    super().__init__(name=agent.name, description=agent.description)

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    data['name'] = data['agent'].name
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    from ..agents.llm_agent import LlmAgent

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      result = _automatic_function_calling_util.build_function_declaration(
          func=self.agent.input_schema, variant=self._api_variant
      )
    else:
      result = types.FunctionDeclaration(
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  'request': types.Schema(
                      type=types.Type.STRING,
                  ),
              },
              required=['request'],
          ),
          description=self.agent.description,
          name=self.name,
      )
    result.name = self.name
    return result

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    from ..agents.llm_agent import LlmAgent

    if self.skip_summarization:
      tool_context.actions.skip_summarization = True

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      input_value = self.agent.input_schema.model_validate(args)
    else:
      input_value = args['request']

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      if isinstance(input_value, dict):
        input_value = self.agent.input_schema.model_validate(input_value)
      if not isinstance(input_value, self.agent.input_schema):
        raise ValueError(
            f'Input value {input_value} is not of type'
            f' `{self.agent.input_schema}`.'
        )
      content = types.Content(
          role='user',
          parts=[
              types.Part.from_text(
                  text=input_value.model_dump_json(exclude_none=True)
              )
          ],
      )
    else:
      content = types.Content(
          role='user',
          parts=[types.Part.from_text(text=input_value)],
      )
    runner = Runner(
        app_name=self.agent.name,
        agent=self.agent,
        # TODO(kech): Remove the access to the invocation context.
        #   It seems we don't need re-use artifact_service if we forward below.
        artifact_service=tool_context._invocation_context.artifact_service,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    session = runner.session_service.create_session(
        app_name=self.agent.name,
        user_id='tmp_user',
        state=tool_context.state.to_dict(),
    )

    last_event = None
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
      # Forward state delta to parent session.
      if event.actions.state_delta:
        tool_context.state.update(event.actions.state_delta)
      last_event = event

    if runner.artifact_service:
      # Forward all artifacts to parent session.
      for artifact_name in runner.artifact_service.list_artifact_keys(
          app_name=session.app_name,
          user_id=session.user_id,
          session_id=session.id,
      ):
        if artifact := runner.artifact_service.load_artifact(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id,
            filename=artifact_name,
        ):
          tool_context.save_artifact(filename=artifact_name, artifact=artifact)

    if (
        not last_event
        or not last_event.content
        or not last_event.content.parts
        or not last_event.content.parts[0].text
    ):
      return ''
    if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
      tool_result = self.agent.output_schema.model_validate_json(
          last_event.content.parts[0].text
      ).model_dump(exclude_none=True)
    else:
      tool_result = last_event.content.parts[0].text
    return tool_result
