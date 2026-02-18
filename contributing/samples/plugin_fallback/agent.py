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

"""Sample agent demonstrating FallbackPlugin usage.

This sample shows how to use the FallbackPlugin to automatically track and
annotate model fallback events when the primary model returns a retriable
error (e.g. HTTP 429 rate-limit or 504 gateway timeout).

The plugin does NOT re-issue the request itself. For the actual retry to happen
you should pair it with a model that has built-in fallback support, such as
LiteLlm's `fallbacks` parameter (see contributing/samples/litellm_with_fallback_models).

Usage:
  adk run contributing/samples/plugin_fallback

The agent will respond to any prompt. When running against real models you
can observe the fallback metadata written to LlmResponse.custom_metadata
whenever a 429 or 504 error is returned by the primary model.
"""

import random
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.fallback_plugin import FallbackPlugin


def roll_die(sides: int) -> dict[str, Any]:
  """Roll a die and return the result.

  Args:
    sides: The number of sides on the die.

  Returns:
    A dictionary with the die result.
  """
  if sides < 2:
    return {"error": f"A die must have at least 2 sides, got {sides}."}
  result = random.randint(1, sides)
  return {"sides": sides, "result": result}


root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="fallback_demo_agent",
    description="A simple agent that demonstrates the FallbackPlugin.",
    instruction="""
      You are a helpful assistant. When asked to roll a die, use the
      roll_die tool with the requested number of sides and report the result.
    """,
    tools=[roll_die],
)

fallback_plugin = FallbackPlugin(
    root_model="gemini-3-flash-preview",
    fallback_model="gemini-2.5-pro",
    error_status=[429, 504],
)

app = App(
    agent=root_agent,
    name="plugin_fallback_demo",
    plugins=[fallback_plugin],
)
