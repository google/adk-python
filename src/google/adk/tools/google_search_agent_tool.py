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
from typing import Union

from google.genai import types
from typing_extensions import override

from ..agents.llm_agent import LlmAgent
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..models.base_llm import BaseLlm
from ..utils.context_utils import Aclosing
from ._forwarding_artifact_service import ForwardingArtifactService
from .agent_tool import AgentTool
from .google_search_tool import google_search
from .tool_context import ToolContext


def create_google_search_agent(model: Union[str, BaseLlm]) -> LlmAgent:
  """Create a sub-agent that only uses google_search tool."""
  return LlmAgent(
      name='google_search_agent',
      model=model,
      description=(
          'An agent for performing Google search using the `google_search` tool'
      ),
      instruction="""
        You are a specialized Google search agent.

        When given a search query, use the `google_search` tool to find the related information.
      """,
      tools=[google_search],
  )


class GoogleSearchAgentTool(AgentTool):
  """A tool that wraps a sub-agent that only uses google_search tool.

  This is a workaround to support using google_search tool with other tools.
  TODO(b/448114567): Remove once the workaround is no longer needed.

  Attributes:
    model: The model to use for the sub-agent.
  """

  def __init__(self, agent: LlmAgent):
    self.agent = agent
    super().__init__(agent=self.agent)

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    from ..agents.llm_agent import LlmAgent
    from ..runners import Runner
    from ..sessions.in_memory_session_service import InMemorySessionService

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      input_value = self.agent.input_schema.model_validate(args)
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
          parts=[types.Part.from_text(text=args['request'])],
      )
    runner = Runner(
        app_name=self.agent.name,
        agent=self.agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=tool_context._invocation_context.credential_service,
        plugins=list(tool_context._invocation_context.plugin_manager.plugins),
    )

    state_dict = {
        k: v
        for k, v in tool_context.state.to_dict().items()
        if not k.startswith('_adk')  # Filter out adk internal states
    }
    session = await runner.session_service.create_session(
        app_name=self.agent.name,
        user_id=tool_context._invocation_context.user_id,
        state=state_dict,
    )

    last_content = None
    last_grounding_metadata = None
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        )
    ) as agen:
      async for event in agen:
        # Forward state delta to parent session.
        if event.actions.state_delta:
          tool_context.state.update(event.actions.state_delta)
        if event.content:
          last_content = event.content
          last_grounding_metadata = event.grounding_metadata

    if not last_content:
      return ''
    merged_text = '\n'.join(p.text for p in last_content.parts if p.text)
    if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
      tool_result = self.agent.output_schema.model_validate_json(
          merged_text
      ).model_dump(exclude_none=True)
    else:
      tool_result = merged_text

    if last_grounding_metadata:
      tool_context.state['temp:_adk_grounding_metadata'] = (
          last_grounding_metadata
      )
    return tool_result
