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

"""ADK utils for a LLMAgent interacting with a simulation environment."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

from absl import logging
from google.adk import runners
from google.adk.agents import base_agent
from google.adk.agents import llm_agent
from google.adk.agents import loop_agent
from google.adk.events import event as event_lib
from google.adk.models import google_llm
from google.adk.planners import built_in_planner
from google.adk.tools import base_tool
from google.genai import types
from retry import api as retry


class EnvResponse(Protocol):
  """Environment response protocol."""

  observation: str
  done: bool
  reward: float


@runtime_checkable
class Env(Protocol):
  """Environment protocol."""

  def step(self, action: types.Part) -> EnvResponse:
    """Steps the environment with the given action."""
    ...

  def reset(self, task_index: int) -> EnvResponse:
    """Resets the environment to the given task index."""
    ...


class _Tool(base_tool.BaseTool):
  """A tool that executes an action in the environment."""

  class Config:
    arbitrary_types_allowed = True

  def __init__(
      self,
      function_declaration: types.FunctionDeclaration,
      env: Env,
  ):
    """Initializes the tool.

    Args:
      function_declaration: The function declaration of the tool.
      env: The environment to interact with.
    """
    super().__init__(
        name=function_declaration.name,
        description=function_declaration.description,
    )
    self._function_declaration = function_declaration
    self._env = env

  def _get_declaration(self) -> types.FunctionDeclaration:
    return self._function_declaration

  async def run_async(self, *, args: Dict[str, Any], tool_context: Any) -> str:
    """Runs the tool by converting tool call to env action and stepping env."""
    env_response = self._env.step(
        types.Part(function_call=types.FunctionCall(name=self.name, args=args))
    )
    # We modify the ADK session state with the updates from the environment,
    # in particular `done` and `reward`. These can be consumed downstream for
    # instance to extract the trajectory reward or interrupt the loop.
    tool_context.actions.state_delta['done'] = env_response.done
    tool_context.actions.state_delta['reward'] = env_response.reward
    tool_context.actions.skip_summarization = True
    if env_response.done:
      tool_context.actions.escalate = True
    return env_response.observation


def _default_retry_options() -> types.HttpRetryOptions:
  return types.HttpRetryOptions(
      initial_delay=2,
      attempts=4,
      max_delay=None,
      exp_base=2.0,
  )


def _adk_agent(
    instruction: str,
    tools: list[base_tool.BaseTool],
    temperature: float,
    model: str | None = None,
    name: str | None = None,
) -> llm_agent.LlmAgent:
  """Creates an ADK LLM agent with the given instruction and tools.

  Args:
    instruction: The instruction for the agent.
    tools: The tools for the agent to use.
    temperature: The temperature for the LLM.
    model: Model to use with the ADK LLMAgent ; defaults to `gemini-2.5-flash`.
    name: Name to set for the ADK LLM agent.

  Returns:
    An ADK LLM agent.
  """
  # TDOO - Allow more flexibility in configuring the agent used in the loop.
  return llm_agent.LlmAgent(
      name=name or 'agent',
      model=google_llm.Gemini(
          model=model or 'gemini-2.5-flash',
          retry_options=_default_retry_options(),
      ),
      planner=built_in_planner.BuiltInPlanner(
          thinking_config=types.ThinkingConfig(
              thinking_budget=-1, include_thoughts=False
          )
      ),
      instruction=instruction,
      tools=tools,
      generate_content_config=types.GenerateContentConfig(
          temperature=temperature,
          tool_config=types.ToolConfig(
              function_calling_config=types.FunctionCallingConfig(
                  mode=types.FunctionCallingConfigMode.VALIDATED
              )
          ),
          http_options=types.HttpOptions(
              timeout=30000,
              retry_options=_default_retry_options(),
          ),
      ),
  )


class _UserAgent(base_agent.BaseAgent):
  """An agent that wraps the provided environment and simulates an user."""

  env: Env

  class Config:
    arbitrary_types_allowed = True

  async def _run_async_impl(self, ctx: Any) -> Any:
    """Runs the user agent."""
    if not ctx.session.events:
      raise ValueError(
          'No prior session events, this is unexpected as the user agent cannot'
          ' be the first step in the interaction loop.'
      )
    last_event = ctx.session.events[-1]

    # Function tool
    if last_event.content and last_event.content.role == 'user':
      return

    if last_event.content and last_event.content.parts:
      next_message = '\n\n'.join([p.text for p in last_event.content.parts])
    else:
      logging.warn('Empty content with event=%s', last_event)
      next_message = ''
    env_response = retry.retry_call(
        self.env.step,
        fargs=(types.Part(text=next_message),),
        tries=3,
        delay=2,
        backoff=2,
    )

    output_event = event_lib.Event(
        content=types.Content(
            parts=[types.Part(text=env_response.observation)], role='user'
        ),
        author='user',
    )
    if env_response.done:
      output_event.actions.escalate = True
    output_event.actions.state_delta['reward'] = env_response.reward
    output_event.actions.state_delta['done'] = env_response.done
    yield output_event


def run_environment_loop(
    instruction: str,
    env: Env,
    temperature: float,
    tools: list[types.FunctionDeclaration],
    task_index: int,
    max_num_steps: int = 30,
    plugins: Optional[Any] = None,
    agent_model: str | None = None,
    agent_name: str | None = None,
) -> Generator[event_lib.Event]:
  """Defines and runs an ADK LLM Agent in the provided simulation environment.

  Args:
    instruction: The instruction for the agent.
    env: The environment to interact with.
    temperature: The temperature for the LLM.
    tools: The tools for the agent to use.
    task_index: The index of the task to run.
    max_num_steps: The maximum number of steps to run LLM agent - environment
      interaction loop.
    plugins: Optional plugins to use in the runner.
    agent_model: Model to use with the ADK LLMAgent ; defaults to
      `gemini-2.5-flash`.
    agent_name: Name to set for the ADK LLM agent.

  Returns:
    A generator of events from the agent run.

  Yields:
    All the events from the environment loop including:
      - Initial message from environment reset
      - LLMAgent generated text and function calls
      - Environment tools / users generated text responses
      - Environment user
  """
  # We use an agent loop to orchestrate the llm-agent and the environment
  # interactions. In particular to:
  # - ensure that LLMAgent and environment / user are called one after the
  # other
  # - the number of interaction steps is pre-defined (early exit is possible).
  agent = loop_agent.LoopAgent(
      name='env_loop_agent',
      max_iterations=max_num_steps,
      sub_agents=[
          _adk_agent(
              instruction=instruction,
              tools=[_Tool(t, env) for t in tools],
              temperature=temperature,
              model=agent_model,
              name=agent_name,
          ),
          _UserAgent(
              name='user_agent',
              env=env,
          ),
      ],
  )

  async def _async_run():
    runner = runners.InMemoryRunner(
        agent=agent,
        app_name='eval_app',
        plugins=plugins,
    )
    session = await runner.session_service.create_session(
        app_name='eval_app', user_id='eval_user'
    )
    env_reset_res = env.reset(task_index=task_index)
    initial_message = types.Content(
        role='user', parts=[types.Part(text=env_reset_res.observation)]
    )
    # The initial message is generated by the environment `reset` within the
    # implementation of this function - as the first step of the trace.
    # We yield this first step to ensure we provide a full trace to the user.
    events = [
        event_lib.Event(
            author='user',
            content=initial_message,
        )
    ]
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=initial_message,
    ):
      events.append(event)
    return events

  return asyncio.run(_async_run())
