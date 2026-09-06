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

"""TealTiger governance plugin for ADK.

Wraps ``tealtiger.integrations.google_adk.TealTigerCallback`` so its tool
hooks run as an App-level plugin (every agent under the App) instead of
per-agent ``before_tool_callback`` / ``after_tool_callback`` fields.

Install the SDK::

    pip install tealtiger

Usage::

    from google.adk.apps.app import App
    from google.adk.plugins import TealTigerPlugin

    app = App(
        name="my_app",
        root_agent=root_agent,
        plugins=[
            TealTigerPlugin(
                policies=[
                    {"type": "pii_block", "categories": ["ssn", "credit_card"]},
                    {"type": "cost_limit", "max_per_session": 5.00},
                ],
                mode="ENFORCE",
            )
        ],
    )
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import inspect
import logging
from typing import Any

from typing_extensions import override

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)

_EVALUATION_FAILED_CONTENT = (
    "[GOVERNANCE DENIED] TealTiger evaluation failed (tool={tool_name})"
)


class TealTigerPlugin(BasePlugin):
  """ADK plugin that enforces TealTiger governance before and after tools.

  Delegates to ``TealTigerCallback``:

  * ``before_tool_callback`` — PII, secrets, tool allowlist, cost budget,
    and freeze, using the callback's ``mode`` (``OBSERVE``, ``MONITOR``,
    or ``ENFORCE``).
  * ``after_tool_callback`` — records the execution outcome for audit.

  A non-``None`` dict from ``before_tool`` short-circuits the tool. TealTiger
  only denies in ``ENFORCE`` mode.
  """

  def __init__(
      self,
      *,
      policies: Sequence[Mapping[str, Any]] | None = None,
      mode: str = "OBSERVE",
      agent_id: str | None = None,
      on_decision: Callable[[dict[str, Any]], Any] | None = None,
      model: str = "gemini-3.6-flash",
      cost_per_tool_call: float = 0.0015,
      block_on_evaluation_failure: bool = True,
      name: str = "tealtiger_plugin",
      callback: Any | None = None,
  ) -> None:
    """Initializes the TealTiger governance plugin.

    Args:
      policies: Governance policy dicts forwarded to
        ``TealTigerCallback`` (``pii_block``, ``tool_allowlist``,
        ``secret_detection``, ``cost_limit``). Ignored when ``callback``
        is provided.
      mode: ``OBSERVE``, ``MONITOR``, or ``ENFORCE``. Only ``ENFORCE``
        blocks denied tool calls. Ignored when ``callback`` is provided.
      agent_id: Agent identifier for audit correlation.
      on_decision: Optional callback invoked with each governance
        decision dict.
      model: Model name used for tool-cost estimates.
      cost_per_tool_call: Fallback USD cost when model pricing is
        unavailable.
      block_on_evaluation_failure: If True (default), treat SDK errors as
        a deny so ungoverned tool calls do not proceed.
      name: Plugin instance name used in logs.
      callback: Optional pre-constructed ``TealTigerCallback`` (e.g. for
        testing). If ``None``, one is constructed, which requires the
        TealTiger SDK.
    """
    super().__init__(name)
    self._block_on_evaluation_failure = block_on_evaluation_failure
    if callback is not None:
      self._callback = callback
    else:
      self._callback = _build_default_callback(
          policies=policies,
          mode=mode,
          agent_id=agent_id,
          on_decision=on_decision,
          model=model,
          cost_per_tool_call=cost_per_tool_call,
      )

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    """Run TealTiger ``before_tool`` and short-circuit when it denies.

    Returns:
      The dict returned by TealTiger to skip the tool, or ``None`` to
      proceed.
    """
    try:
      result = await _await_if_needed(
          self._callback.before_tool(
              callback_context=tool_context,
              tool=tool,
              args=tool_args,
              tool_context=tool_context,
          )
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception(
          "TealTiger before_tool evaluation failed (tool=%s)", tool.name
      )
      if self._block_on_evaluation_failure:
        return {
            "content": _EVALUATION_FAILED_CONTENT.format(tool_name=tool.name)
        }
      return None
    if result is None:
      return None
    if isinstance(result, dict):
      return result
    logger.warning(
        "TealTiger before_tool returned a non-dict (%s); allowing the tool",
        type(result).__name__,
    )
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Run TealTiger ``after_tool`` for audit. Never replaces the result."""
    try:
      await _await_if_needed(
          self._callback.after_tool(
              callback_context=tool_context,
              tool=tool,
              args=tool_args,
              tool_context=tool_context,
              result=result,
          )
      )
    except Exception:  # pylint: disable=broad-except
      logger.exception("TealTiger after_tool audit failed (tool=%s)", tool.name)
    return None

  @override
  async def close(self) -> None:
    """Closes the underlying callback if it exposes ``close``."""
    close = getattr(self._callback, "close", None)
    if close is None:
      return
    await _await_if_needed(close())

  def freeze(self) -> None:
    """Freeze the agent so TealTiger denies subsequent tool calls."""
    freeze = getattr(self._callback, "freeze", None)
    if freeze is not None:
      freeze()

  def unfreeze(self) -> None:
    """Clear a freeze and restore normal governance."""
    unfreeze = getattr(self._callback, "unfreeze", None)
    if unfreeze is not None:
      unfreeze()

  @property
  def decisions(self) -> list[dict[str, Any]]:
    """Governance decisions recorded by TealTiger, if available."""
    recorded = getattr(self._callback, "decisions", None)
    if recorded is None:
      return []
    return list(recorded)

  @property
  def deny_count(self) -> int:
    """Count of denied tool calls, if TealTiger exposes it."""
    count = getattr(self._callback, "deny_count", None)
    if count is None:
      return 0
    return int(count)

  @property
  def total_cost(self) -> float:
    """Cumulative cost tracked by TealTiger, if available."""
    cost = getattr(self._callback, "total_cost", None)
    if cost is None:
      return 0.0
    return float(cost)


def _build_default_callback(
    *,
    policies: Sequence[Mapping[str, Any]] | None,
    mode: str,
    agent_id: str | None,
    on_decision: Callable[[dict[str, Any]], Any] | None,
    model: str,
    cost_per_tool_call: float,
) -> Any:
  try:
    from tealtiger.integrations.google_adk import TealTigerCallback  # type: ignore
  except ImportError as e:
    raise ImportError(
        "TealTiger is not installed. Run: pip install tealtiger"
    ) from e
  policy_list = [dict(policy) for policy in policies] if policies else None
  return TealTigerCallback(
      policies=policy_list,
      mode=mode,
      agent_id=agent_id,
      on_decision=on_decision,
      model=model,
      cost_per_tool_call=cost_per_tool_call,
  )


async def _await_if_needed(value: Any) -> Any:
  if inspect.isawaitable(value):
    return await value
  return value
