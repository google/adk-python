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

"""A before_tool_callback that gates tool calls on a classifier's answer."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
import logging
from typing import Any
from typing import Literal

from google.adk.labs.classifiers import BaseClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import ClassifierInput
from google.adk.labs.classifiers import JSONContent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

StateExtractor = Callable[
    [BaseTool, dict[str, Any], ToolContext], ClassifierInput
]
ToolGateCallback = Callable[
    [BaseTool, dict[str, Any], ToolContext], Awaitable[dict[str, Any] | None]
]


def _default_state(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> ClassifierInput:
  del tool_context  # Unused.
  return {"tool": tool.name, "description": tool.description, "args": args}


def create_tool_gate_callback(
    *,
    classifier: BaseClassifier,
    instructions: JSONContent,
    criteria: JSONContent | None = None,
    threshold: float = 0.5,
    on_threshold_exceeded: Literal["block", "confirm"] = "block",
    state_extractor: StateExtractor | None = None,
) -> ToolGateCallback:
  """Creates a before_tool_callback that asks a classifier if a call is risky.

  Before each tool call, the callback asks `classifier` a `Binary` question
  about the call. If the probability of true is at or above `threshold`, the
  call is blocked, or held until the user confirms it.

  Args:
    classifier: Any `BaseClassifier`, such as `LlmClassifier` or a dedicated
      decision model.
    instructions: The yes/no question to ask about each call, where yes means
      the call is risky.
    criteria: Optionally, a description of what counts as risky.
    threshold: The probability of risky at or above which the gate acts.
    on_threshold_exceeded: "block" skips the call and returns an error to the
      model. "confirm" asks the user through `tool_context.request_confirmation`
      and runs the call only if they approve.
    state_extractor: Builds the classifier input from the tool, its arguments
      and the tool context. Defaults to the tool's name, description and
      arguments.

  Returns:
    An async callback to pass as an agent's `before_tool_callback`.
  """
  if not 0.0 <= threshold <= 1.0:
    raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
  if on_threshold_exceeded not in ("block", "confirm"):
    raise ValueError(
        "on_threshold_exceeded must be 'block' or 'confirm', got"
        f" {on_threshold_exceeded!r}."
    )
  question = Binary(instructions=instructions, criteria=criteria)
  extract_state = state_extractor or _default_state

  async def tool_gate(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any] | None:
    # A confirmed call comes back through this callback with the user's
    # answer attached; classifying it again could ask the user twice.
    confirmation = tool_context.tool_confirmation
    if confirmation is not None:
      if confirmation.confirmed:
        return None
      return {"error": f"The user rejected this call to {tool.name}."}

    try:
      response = await classifier.classify(
          input=extract_state(tool, args, tool_context),
          questions={"risky": question},
      )
      answer = response.answers["risky"]
      if not isinstance(answer, BinaryAnswer):
        raise ValueError(f"Expected a BinaryAnswer, got {answer!r}.")
      probability = answer.probability
    except Exception:  # pylint: disable=broad-exception-caught
      # Fail closed: a classifier outage must not let risky calls through.
      logger.exception("Tool gate classifier failed for %s.", tool.name)
      probability = 1.0

    logger.info("Tool gate: %s risk=%.2f", tool.name, probability)
    if probability < threshold:
      return None

    if on_threshold_exceeded == "confirm":
      tool_context.request_confirmation(
          hint=(
              f"The tool gate flagged this call to {tool.name} as risky"
              f" ({probability:.0%}). Approve it?"
          ),
      )
      # Pause for the user instead of letting the model react to the error,
      # which would make it call the tool again.
      tool_context.actions.skip_summarization = True
      return {
          "error": (
              "This tool call requires confirmation, please approve or reject."
          )
      }
    return {
        "error": (
            f"The call to {tool.name} was blocked because it was classified"
            " as risky. Do not retry it; tell the user it was blocked."
        )
    }

  return tool_gate
