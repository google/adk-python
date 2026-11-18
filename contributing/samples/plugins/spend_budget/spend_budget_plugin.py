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

"""A plugin that caps the USD spend of each agent run and of the whole run."""

from __future__ import annotations

from collections.abc import Mapping
import logging

from agent_budget import Budget
from agent_budget import BudgetError
from agent_budget import Reservation
from agent_budget import Scope
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

logger = logging.getLogger("google_adk." + __name__)

SPEND_BUDGET_EXCEEDED = "SPEND_BUDGET_EXCEEDED"
"""`LlmResponse.error_code` of a model call the plugin refused."""


class SpendBudgetPlugin(BasePlugin):
  """Caps the USD spend of each agent run and of the whole run.

  Each `Runner.run_async` call gets a budget of `run_usd`. Each agent run
  inside it (each workflow node run, keyed by its node path, so a fan-out of
  one agent definition gets one budget per node) gets a child budget of
  `agent_usd`, or the entry for its name in `agent_usd_overrides`, taken out of
  the run budget when the agent makes its first model call. What an agent run
  does not spend goes back to the run budget when it ends.

  Before each model call the plugin reserves an estimate of its cost (prompt
  characters / 4 input tokens plus `max_output_tokens`, or
  `default_output_tokens`, output tokens). If the reservation does not fit,
  the call is not made: the plugin returns an `LlmResponse` with error code
  `SPEND_BUDGET_EXCEEDED`, which ends that agent's turn while the rest of the
  run goes on. After the call, the reservation is replaced by the cost of the
  real `usage_metadata`. Because the estimate is held before the call, agents
  running in parallel cannot together start more than the budget allows.
  """

  def __init__(
      self,
      *,
      input_usd_per_million: float,
      output_usd_per_million: float,
      run_usd: float,
      agent_usd: float | None = None,
      agent_usd_overrides: Mapping[str, float] | None = None,
      default_output_tokens: int = 4096,
      name: str = "spend_budget",
  ) -> None:
    """Initializes the plugin.

    Args:
      input_usd_per_million: Price of one million input tokens, in USD.
      output_usd_per_million: Price of one million output (and thinking)
        tokens, in USD.
      run_usd: Cap for one invocation (one `Runner.run_async` call).
      agent_usd: Cap for each agent run. None leaves agent runs limited only
        by the run budget.
      agent_usd_overrides: Caps for agent runs by agent name, taking precedence
        over `agent_usd`.
      default_output_tokens: Output tokens to reserve for a request that does
        not set `max_output_tokens`.
      name: The plugin name.
    """
    super().__init__(name=name)
    self.input_usd_per_million = input_usd_per_million
    self.output_usd_per_million = output_usd_per_million
    self.run_usd = run_usd
    self.agent_usd = agent_usd
    self.agent_usd_overrides = dict(agent_usd_overrides or {})
    self.default_output_tokens = default_output_tokens
    self._runs: dict[str, Budget] = {}
    self._scopes: dict[tuple[str, str], Scope] = {}
    # Keyed by the identity of the model response event's actions, which the
    # before and after model callbacks of one call share.
    self._in_flight: dict[int, tuple[str, EventActions, Reservation]] = {}

  def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
    """Returns the price of a call. Override for per-model or cached rates."""
    return (
        input_tokens * self.input_usd_per_million
        + output_tokens * self.output_usd_per_million
    ) / 1_000_000

  def estimate_usd(self, llm_request: LlmRequest) -> float:
    """Returns the amount to reserve before `llm_request` is sent."""
    chars = sum(
        len(part.text or "")
        for content in llm_request.contents
        for part in content.parts or []
    )
    config = llm_request.config
    if config and isinstance(config.system_instruction, str):
      chars += len(config.system_instruction)
    output_tokens = (
        config.max_output_tokens
        if config and config.max_output_tokens
        else self.default_output_tokens
    )
    return self.cost_usd(chars // 4 + 1, output_tokens)

  def _agent_limit(self, agent_name: str) -> float | None:
    return self.agent_usd_overrides.get(agent_name, self.agent_usd)

  def _run_budget(self, invocation_id: str) -> Budget:
    run = self._runs.get(invocation_id)
    if run is None:
      run = Budget(self.run_usd, name="run")
      self._runs[invocation_id] = run
    return run

  def _close_scope(self, key: tuple[str, str]) -> None:
    scope = self._scopes.pop(key, None)
    if scope is not None:
      receipt = scope.close()
      logger.info(
          "agent run %s spent $%.6f of $%.6f",
          receipt.scope,
          receipt.spent_usd,
          receipt.held_usd,
      )

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    self._run_budget(invocation_context.invocation_id)

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    invocation_id = callback_context.invocation_id
    agent_name = callback_context.agent_name
    run_key = callback_context.node_path or agent_name
    budget = self._run_budget(invocation_id)
    try:
      limit = self._agent_limit(agent_name)
      scope = self._scopes.get((invocation_id, run_key))
      if scope is None and limit is not None and budget.remaining_usd > 0:
        # An agent run gets its cap, or what is left of the run if less.
        scope = budget.child(run_key, usd=min(limit, budget.remaining_usd))
        self._scopes[(invocation_id, run_key)] = scope
      # With nothing left in the run, reserving on the run budget refuses.
      reservation = (scope or budget).reserve(self.estimate_usd(llm_request))
    except BudgetError as e:
      logger.warning("refused a model call of %s: %s", run_key, e)
      message = f"Agent '{agent_name}' stopped: {e}"
      return LlmResponse(
          content=types.Content(role="model", parts=[types.Part(text=message)]),
          error_code=SPEND_BUDGET_EXCEEDED,
          error_message=message,
      )
    actions = callback_context.actions
    self._in_flight[id(actions)] = (invocation_id, actions, reservation)
    return None

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    if llm_response.partial:
      return None
    entry = self._in_flight.pop(id(callback_context.actions), None)
    if entry is None:
      return None
    reservation = entry[2]
    usage = llm_response.usage_metadata
    if usage is None:
      # No usage reported: book the estimate rather than nothing.
      reservation.settle(reservation.estimate_usd)
      return None
    input_tokens = (usage.prompt_token_count or 0) + (
        usage.tool_use_prompt_token_count or 0
    )
    output_tokens = (usage.candidates_token_count or 0) + (
        usage.thoughts_token_count or 0
    )
    reservation.settle(self.cost_usd(input_tokens, output_tokens))
    return None

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> LlmResponse | None:
    entry = self._in_flight.pop(id(callback_context.actions), None)
    if entry is not None:
      entry[2].release()
    return None

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    self._close_scope((
        callback_context.invocation_id,
        callback_context.node_path or callback_context.agent_name,
    ))

  async def on_agent_error_callback(
      self,
      *,
      agent: BaseAgent,
      callback_context: CallbackContext,
      error: Exception,
  ) -> None:
    await self.after_agent_callback(
        agent=agent, callback_context=callback_context
    )

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    self._end_run(invocation_context.invocation_id)

  async def on_run_error_callback(
      self, *, invocation_context: InvocationContext, error: Exception
  ) -> None:
    self._end_run(invocation_context.invocation_id)

  def _end_run(self, invocation_id: str) -> None:
    for call_id, entry in list(self._in_flight.items()):
      if entry[0] == invocation_id:
        del self._in_flight[call_id]
        entry[2].release()
    for key in [k for k in self._scopes if k[0] == invocation_id]:
      self._close_scope(key)
    run = self._runs.pop(invocation_id, None)
    if run is not None:
      logger.info("run spent $%.6f of $%.6f", run.spent_usd, run.limit_usd)
