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

"""Agent-lifecycle callbacks backed by TypeSafe judgments.

Each factory here returns a plain callback usable as
``LlmAgent(before_model_callback=...)`` or
``LlmAgent(before_tool_callback=...)``, so a TypeSafe judgment can run
automatically at a lifecycle point instead of through a model-initiated tool
call.
"""

from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Optional

from ...agents.callback_context import CallbackContext
from ...agents.llm_agent import BeforeModelCallback
from ...agents.llm_agent import BeforeToolCallback
from ...models.llm_request import LlmRequest
from ...tools.base_tool import BaseTool
from ...tools.tool_context import ToolContext
from ._typesafe_classifier import Choice
from ._typesafe_classifier import Noul
from ._typesafe_classifier import TypesafeClassifier

__all__ = ["create_model_router_callback", "create_tool_gate_callback"]


def _latest_user_text(llm_request: LlmRequest) -> Optional[str]:
  """Returns the text of the most recent user turn, if any."""
  for content in reversed(llm_request.contents or []):
    if content.role != "user":
      continue
    texts = [part.text for part in content.parts or [] if part.text]
    if texts:
      return "\n".join(texts)
  return None


def create_model_router_callback(
    *,
    classifier: TypesafeClassifier,
    criteria: Mapping[str, Optional[str]],
    model_map: Mapping[str, str],
    instructions: str = "Which model should handle this request?",
) -> BeforeModelCallback:
  """Builds a `before_model_callback` that routes based on the latest user message.

  Classifies the latest user message with a `Choice` question and, on a match
  in `model_map`, sends the request to that model instead.

  Args:
    classifier: The classifier used to reach TypeSafe's Jev model.
    criteria: Choice options mapped to their descriptions, e.g. `{"simple":
      "a short factual question", "complex": "a multi-step task"}`.
    model_map: Maps a `criteria` key to the ADK model name to route to.
    instructions: The question put to Jev.

  Returns:
    A callback usable as `LlmAgent(before_model_callback=...)`.
  """
  choice_question = Choice(instructions=instructions, criteria=criteria)

  async def _route_by_classification(
      callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    del callback_context  # Unused: routing only needs the request itself.
    state = _latest_user_text(llm_request)
    if state is None:
      return None
    response = await classifier.classify(
        state=state, questions={"route": choice_question}
    )
    chosen_model = model_map.get(response.answers["route"].choice)
    if chosen_model:
      llm_request.model = chosen_model
    return None

  return _route_by_classification


def create_tool_gate_callback(
    *,
    classifier: TypesafeClassifier,
    instructions: str,
    threshold: float = 0.5,
    criteria: Optional[Mapping[str, Optional[str]]] = None,
) -> BeforeToolCallback:
  """Builds a `before_tool_callback` that blocks risky tool calls.

  Classifies the pending tool call with a `Noul` question and blocks it when
  the risk probability meets or exceeds `threshold`.

  Args:
    classifier: The classifier used to reach TypeSafe's Jev model.
    instructions: The yes/no question put to Jev, e.g. "Could this tool call
      cause irreversible harm (delete data, send money, send a message)?".
    threshold: Block the call when its risk probability is >= this value.
    criteria: Optional descriptions for the `true`/`false` answers.

  Returns:
    A callback usable as `LlmAgent(before_tool_callback=...)`.
  """
  risk_question = Noul(instructions=instructions, criteria=criteria)

  async def _gate_risky_tool_calls(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> Optional[dict[str, Any]]:
    del tool_context  # Unused: the judgment only needs the call itself.
    response = await classifier.classify(
        state={"tool": tool.name, "args": args},
        questions={"risk": risk_question},
    )
    risk = response.answers["risk"].noul
    if risk >= threshold:
      return {
          "error": (
              f"Blocked by TypeSafe risk gate: {tool.name} scored"
              f" {risk:.2f} risk (threshold {threshold})."
          )
      }
    return None

  return _gate_risky_tool_calls
