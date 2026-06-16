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

"""A security guardrail plugin backed by Agent Threat Rules (ATR).

ATR (https://github.com/Agent-Threat-Rule/agent-threat-rules) is an open,
MIT-licensed detection ruleset for AI-agent threats such as prompt injection,
instruction override, and data exfiltration. This sample wires the `pyatr`
engine into ADK's plugin callbacks so that a single plugin enforces policy
*globally* across every agent, model call, and tool call managed by a Runner.

Install the engine before running:

    pip install pyatr

Enforcement points (each one short-circuits the rest of the lifecycle):
  * `before_run_callback`   -- halts the run on a malicious user message.
  * `before_model_callback` -- skips the model call if the assembled prompt
    still carries a threat (defense in depth, e.g. injected tool/context text).
  * `before_tool_callback`  -- fails closed: returns an error dict instead of
    executing a tool whose arguments match a rule.
"""

from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# pyatr is an optional, third-party engine (`pip install pyatr`). Import it
# lazily so this sample module can still be imported for inspection without it.
try:
  from pyatr import scan as _atr_scan
except ImportError:  # pragma: no cover - exercised only without pyatr installed
  _atr_scan = None

# Ordering used to compare a match's severity against `min_severity`.
_SEVERITY_RANK = {
    'info': 0,
    'low': 1,
    'medium': 2,
    'high': 3,
    'critical': 4,
}


def _text_of(content: Optional[types.Content]) -> str:
  """Concatenate the text parts of a `types.Content`."""
  if content is None or not content.parts:
    return ''
  return '\n'.join(part.text for part in content.parts if part.text)


class AtrGuardrailPlugin(BasePlugin):
  """Blocks agent activity that matches an Agent Threat Rules signature."""

  def __init__(self, min_severity: str = 'high') -> None:
    """Initialize the guardrail.

    Args:
      min_severity: The lowest rule severity that should block. One of
        `info`, `low`, `medium`, `high`, `critical`.
    """
    super().__init__(name='atr_guardrail')
    self.min_severity = min_severity
    self._threshold = _SEVERITY_RANK.get(min_severity, 3)

  def _first_block(self, text: str) -> Optional[Any]:
    """Return the highest-severity ATR match at/above the threshold, else None."""
    if _atr_scan is None:
      raise RuntimeError(
          'pyatr is not installed. Run `pip install pyatr` to enable the ATR'
          ' guardrail.'
      )
    if not text.strip():
      return None
    blocking = [
        match
        for match in _atr_scan(text)
        if _SEVERITY_RANK.get(match.severity, 0) >= self._threshold
    ]
    if not blocking:
      return None
    return max(blocking, key=lambda m: _SEVERITY_RANK.get(m.severity, 0))

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Halt the run if the user's message matches a threat rule."""
    match = self._first_block(_text_of(invocation_context.user_content))
    if match is None:
      return None
    print(
        f'[ATR] Blocked user message: rule {match.rule_id} ({match.severity}) -'
        f' {match.title}'
    )
    return types.Content(
        role='model',
        parts=[
            types.Part.from_text(
                text=f'Request blocked by ATR rule {match.rule_id}.'
            )
        ],
    )

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Skip the model call if the assembled prompt still carries a threat."""
    text = '\n'.join(_text_of(content) for content in llm_request.contents)
    match = self._first_block(text)
    if match is None:
      return None
    print(
        f'[ATR] Blocked model request: rule {match.rule_id} ({match.severity})'
        f' - {match.title}'
    )
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[
                types.Part.from_text(
                    text=f'Request blocked by ATR rule {match.rule_id}.'
                )
            ],
        )
    )

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Fail closed: refuse to run a tool whose arguments match a rule."""
    text = '\n'.join(str(value) for value in tool_args.values())
    match = self._first_block(text)
    if match is None:
      return None
    print(
        f'[ATR] Blocked tool `{tool.name}`: rule {match.rule_id}'
        f' ({match.severity}) - {match.title}'
    )
    return {
        'error': f'blocked by ATR rule {match.rule_id}',
        'rule_id': match.rule_id,
        'severity': match.severity,
    }
