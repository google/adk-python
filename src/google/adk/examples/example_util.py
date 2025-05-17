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

"""Utility functions for converting examples to a string that can be used in system instructions in the prompt."""

import logging
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from .base_example_provider import BaseExampleProvider
from .example import Example

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

# Constant parts of the example string
_EXAMPLES_INTRO = (
    "<EXAMPLES>\nBegin few-shot\nThe following are examples of user queries and"
    " model responses using the available tools.\n\n"
)
_EXAMPLES_END = "End few-shot\n<EXAMPLES>"
_EXAMPLE_START = "EXAMPLE {}:\nBegin example\n"
_EXAMPLE_END = "End example\n\n"
_USER_PREFIX = "[user]\n"
_MODEL_PREFIX = "[model]\n"
_FUNCTION_PREFIX = "```\n"
_FUNCTION_CALL_PREFIX = "```tool_code\n"
_FUNCTION_CALL_SUFFIX = "\n```\n"
_FUNCTION_RESPONSE_PREFIX = "```tool_outputs\n"
_FUNCTION_RESPONSE_SUFFIX = "\n```\n"


# TODO(yaojie): Add unit tests for this function.
def convert_examples_to_text(
    examples: list[Example], model: Optional[str]
) -> str:
  """Converts a list of examples to a string that can be used in a system instruction."""
  examples_str = ""
  for example_num, example in enumerate(examples):
    output = f"{_EXAMPLE_START.format(example_num + 1)}{_USER_PREFIX}"
    if example.input and example.input.parts:
      output += (
          "\n".join(part.text for part in example.input.parts if part.text)
          + "\n"
      )

    gemini2 = model is None or "gemini-2" in model
    previous_role = None
    for content in example.output:
      role = _MODEL_PREFIX if content.role == "model" else _USER_PREFIX
      if role != previous_role:
        output += role
      previous_role = role
      for part in content.parts:
        if part.function_call:
          args = []
          # Convert function call part to python-like function call
          for k, v in part.function_call.args.items():
            if isinstance(v, str):
              args.append(f"{k}='{v}'")
            else:
              args.append(f"{k}={v}")
          prefix = _FUNCTION_PREFIX if gemini2 else _FUNCTION_CALL_PREFIX
          output += (
              f"{prefix}{part.function_call.name}({', '.join(args)}){_FUNCTION_CALL_SUFFIX}"
          )
        # Convert function response part to json string
        elif part.function_response:
          prefix = _FUNCTION_PREFIX if gemini2 else _FUNCTION_RESPONSE_PREFIX
          output += f"{prefix}{part.function_response.__dict__}{_FUNCTION_RESPONSE_SUFFIX}"
        elif part.text:
          output += f"{part.text}\n"

    output += _EXAMPLE_END
    examples_str += output

  return f"{_EXAMPLES_INTRO}{examples_str}{_EXAMPLES_END}"


def _get_latest_message_from_user(session: "Session") -> str:
  """Gets the latest message from the user.

  Returns:
    The latest message from the user. If not found, returns an empty string.
  """
  events = session.events
  if not events:
    return ""

  event = events[-1]
  if event.author == "user" and not event.get_function_responses():
    if event.content.parts and event.content.parts[0].text:
      return event.content.parts[0].text
    else:
      logger.warning("No message from user for fetching example.")

  return ""


def build_example_si(
    examples: Union[list[Example], BaseExampleProvider],
    query: str,
    model: Optional[str],
) -> str:
  if isinstance(examples, list):
    return convert_examples_to_text(examples, model)
  if isinstance(examples, BaseExampleProvider):
    return convert_examples_to_text(examples.get_examples(query), model)

  raise ValueError("Invalid example configuration")
