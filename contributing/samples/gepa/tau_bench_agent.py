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

"""Allows to run an ADK agent implementation with a Tau-bench environment.

Note that Tau-bench needs to be installed to run this module. To install
Tau-bench you can follow the steps below:

```
git clone https://github.com/sierra-research/tau-bench.git
cd tau-bench/
pip install -e . --quiet
```
"""
from __future__ import annotations

from typing import Any

import adk_agent
from google.adk.models import llm_response
from google.adk.plugins import base_plugin
from google.genai import types
from tau_bench import envs
from tau_bench import types as tau_bench_types
from tau_bench.agents import tool_calling_agent


class _EnvWrapper:
  """Wraps the Tau-bench environment to match ADK environment protocol."""

  def __init__(self, env: envs.Env):
    self._env = env

  def step(self, action: types.Part) -> adk_agent.EnvResponse:
    if function_call := action.function_call:
      return self._env.step(
          tau_bench_types.Action(
              name=function_call.name, kwargs=function_call.args
          )
      )
    return self._env.step(
        tau_bench_types.Action(
            name=tau_bench_types.RESPOND_ACTION_NAME,
            kwargs=dict(content=action.text),
        )
    )

  def reset(self, task_index: int) -> adk_agent.EnvResponse:
    return self._env.reset(task_index)


def _convert_tool(tool_def: dict[str, Any]) -> types.FunctionDeclaration:
  if tool_def['type'] != 'function':
    raise ValueError(f'Unsupported tool {tool_def}')
  return types.FunctionDeclaration(**tool_def['function'])


_LLM_CALL_ERROR = 'llm_call_error'


class _TauBenchPlugin(base_plugin.BasePlugin):
  """Catches LLM errors and emits event with error code for downstream usage."""

  async def on_model_error_callback(
      self,
      *,
      callback_context: base_plugin.CallbackContext,
      llm_request: base_plugin.LlmRequest,
      error: Exception,
  ) -> llm_response.LlmResponse:
    del callback_context, llm_request  # Unused.
    return llm_response.LlmResponse(
        error_code=_LLM_CALL_ERROR,
        error_message=str(error),
    )


class _ADKAgent(tool_calling_agent.ToolCallingAgent):
  """ADK agent implementation for Tau Bench."""

  def solve(
      self,
      env: envs.Env,
      task_index: int | None = None,
      max_num_steps: int = 30,
  ) -> tau_bench_types.SolveResult:
    """Solves the task using ADK agent.

    Args:
      env: The environment to solve the task in.
      task_index: The index of the task to solve.
      max_num_steps: The maximum number of steps to run the agent.

    Returns:
      The result of the solve.

    Raises:
      - ValueError: If the LLM inference failed.
    """
    # Thought-signature is excluded from the message serialization for the
    # following reasons:
    # - it is not serializable out of the box
    # - it is not relevant for trajectory validation as agent inputs / outputs
    # are.
    content_exclusion = {'parts': {'__all__': 'thought_signature'}}
    messages = [
        types.Content(
            role='system', parts=[types.Part(text=self.wiki)]
        ).model_dump(exclude=content_exclusion),
    ]
    reward = 0.0
    for event in adk_agent.run_environment_loop(
        instruction=self.wiki,
        env=_EnvWrapper(env),
        temperature=self.temperature,
        tools=[_convert_tool(t) for t in env.tools_info],
        task_index=task_index,
        max_num_steps=max_num_steps,
        plugins=[_TauBenchPlugin(name='error_plugin')],
    ):
      if event.error_code == _LLM_CALL_ERROR:
        raise ValueError(f'Error {event.error_code=}: {event.error_message=}')

      if not event.content:
        continue
      messages.append(event.content.model_dump(exclude=content_exclusion))
      reward = event.actions.state_delta.get('reward', reward)
    return tau_bench_types.SolveResult(
        reward=reward,
        info={},
        messages=messages,
    )


# Equivalent of default `agent_factory` from Tau-bench in
#  https://github.com/sierra-research/tau-bench/blob/4754e6b406507dbcbce8e8b3855dcf80aaec18ac/tau_bench/run.py#L124
def adk_agent_factory(
    tools_info: list[dict[str, Any]],
    wiki: str,
    config: tau_bench_types.RunConfig,
) -> tool_calling_agent.ToolCallingAgent:
  """Factory for creating a Tau-bench agent implemented with the ADK.

  Args:
    tools_info: A list of tool definitions.
    wiki: The instructions for the agent.
    config: The run configuration.

  Returns:
    An ADK agent.
  """
  return _ADKAgent(
      tools_info=tools_info,
      wiki=wiki,
      model=config.model,
      provider=config.model_provider,
      temperature=config.temperature,
  )
