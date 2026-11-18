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

"""An ADK agent on an OpenAI reasoning model through the Responses API.

``OpenAIResponsesLlm`` talks to the OpenAI Responses API (``/v1/responses``).
Some reasoning models, such as ``gpt-6-astra``, need it to call tools, because
they reject function tools on Chat Completions. The agent sets a reasoning
effort with ``OpenAIGenerateContentConfig``, and the model's reasoning summaries
come back as thought parts.

``OpenAIResponsesLlm`` reads ``OPENAI_API_KEY`` from the environment (the
openai SDK's default client). Point at a different reasoning model with
``OPENAI_MODEL``. See README.md for details.
"""

from __future__ import annotations

import os
import random

from google.adk import Agent
from google.adk.integrations.openai import OpenAIGenerateContentConfig
from google.adk.integrations.openai import OpenAIResponsesLlm
from google.adk.models.base_llm import BaseLlm


def _build_model() -> BaseLlm:
  """Builds the OpenAI Responses model."""
  # With api_key unset, the default client reads OPENAI_API_KEY. A missing key
  # is reported by the client on the first request, not at import time.
  return OpenAIResponsesLlm(model=os.getenv("OPENAI_MODEL", "gpt-6-astra"))


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime.

  Args:
    nums: The list of numbers to check.

  Returns:
    A str indicating which number is prime.
  """
  if isinstance(nums, int):
    # Tolerate a model passing a single number instead of a list.
    nums = [nums]
  primes = set()
  for number in nums:
    number = int(number)
    if number <= 1:
      continue
    is_prime = True
    for i in range(2, int(number**0.5) + 1):
      if number % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.add(number)
  return (
      "No prime numbers found."
      if not primes
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


root_agent = Agent(
    name="openai_reasoning_agent",
    model=_build_model(),
    description=(
        "A hello-world agent powered by an OpenAI reasoning model that rolls"
        " dice and checks whether numbers are prime."
    ),
    instruction="""
      You are a helpful assistant powered by an OpenAI reasoning model that can
      roll dice and check whether numbers are prime.
      When asked to roll a die, call the roll_die tool with the integer number
      of sides. Never roll a die yourself.
      When asked to check primes, call the check_prime tool with a list of
      integers. Never decide primality yourself.
      When asked to roll a die and then check the result, first call roll_die,
      wait for its result, then call check_prime with that result. Always report
      the number you rolled.
    """,
    tools=[roll_die, check_prime],
    # The reasoning effort tier. Setting it also asks for a concise reasoning
    # summary, which comes back as thought parts. The tiers a model accepts
    # depend on the model and the API; gpt-6-astra accepts low, medium, high,
    # xhigh, and max on the Responses API.
    generate_content_config=OpenAIGenerateContentConfig(effort="medium"),
)
