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

"""Runtime invariants and copy helpers shared by LLM-flow processors."""

from __future__ import annotations

from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar

from google.genai import types
from pydantic import BaseModel

from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.run_config import RunConfig

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent

_ModelT = TypeVar('_ModelT', bound=BaseModel)


def copy_or_none(model: Optional[_ModelT]) -> Optional[_ModelT]:
  """Returns a deep copy of a sub-model that request assembly then mutates."""
  return None if model is None else model.model_copy(deep=True)


def run_config_for_new_live_session(run_config: RunConfig) -> RunConfig:
  """Copies ``run_config`` for a fresh live session, clearing any handle.

  Only ``session_resumption`` is copied. A deep copy of the whole config would
  drag ``http_options`` along, and that can hold a live httpx or aiohttp client
  which raises ``TypeError: cannot pickle``; the rest of the config is not
  mutated here, so sharing it is what the caller wants anyway.
  """
  copied = run_config.model_copy(
      update={'session_resumption': copy_or_none(run_config.session_resumption)}
  )
  if copied.session_resumption:
    copied.session_resumption.handle = None
  return copied


def copy_http_options(
    http_options: types.HttpOptions,
) -> types.HttpOptions:
  """Copies http_options far enough that nothing can write through it.

  Deliberately not a deep copy: the field can carry a live httpx or aiohttp
  client and an SSL context, which raise ``TypeError: cannot pickle`` on a deep
  copy. Those clients are shared on purpose -- a caller supplies one so that
  the SDK uses that exact object.

  Every mutable container is copied instead. Assembly itself only touches
  ``headers``, but a before-model callback is handed the request config and can
  write into any of them, and the source may be the caller's own ``RunConfig``,
  which must not pick up per-request edits.
  """
  updates: dict[str, object] = {}
  for name in ('headers', 'extra_body', 'client_args', 'async_client_args'):
    value = getattr(http_options, name)
    if value is not None:
      updates[name] = dict(value)
  if http_options.retry_options is not None:
    updates['retry_options'] = http_options.retry_options.model_copy()
  return http_options.model_copy(update=updates)


def require_agent(invocation_context: InvocationContext) -> BaseAgent:
  """Returns the agent required by processors that walk the agent tree."""
  agent = invocation_context.agent
  if not isinstance(agent, BaseAgent):
    raise TypeError('LLM flow requires a BaseAgent in InvocationContext.')
  return agent


def as_llm_agent(invocation_context: InvocationContext) -> LlmAgent:
  """Returns the invocation's agent, narrowed to what LLM flows read from it.

  Flows also drive agents defined outside this package that provide the
  LlmAgent surface without subclassing it, so this narrows statically; call
  sites that read an attribute those agents may not define still guard it.
  """
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('LLM flow requires an agent in InvocationContext.')
  return cast('LlmAgent', agent)


def require_agent_name(invocation_context: InvocationContext) -> str:
  """Returns the name shared by agent and workflow-node invocations."""
  agent = invocation_context.agent
  if agent is None:
    raise TypeError('LLM flow requires an agent in InvocationContext.')
  return agent.name


def require_run_config(invocation_context: InvocationContext) -> RunConfig:
  """Returns the run configuration required by model execution."""
  run_config = invocation_context.run_config
  if run_config is None:
    raise ValueError('LLM flow requires a RunConfig in InvocationContext.')
  return run_config
