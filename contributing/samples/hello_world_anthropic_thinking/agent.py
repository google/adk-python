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

from google.adk import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.genai import types


def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime.

  Args:
    nums: The list of numbers to check.

  Returns:
    A str indicating which numbers are prime.
  """
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
      else f"{', '.join(str(num) for num in sorted(primes))} are prime numbers."
  )


root_agent = Agent(
    model=AnthropicLlm(model="claude-sonnet-4-6"),
    name="anthropic_thinking_agent",
    description="An agent that uses Claude extended thinking via Vertex AI.",
    instruction="""
      You are a helpful assistant. Use your reasoning carefully before answering.
      When asked to check prime numbers, use the check_prime tool.
    """,
    tools=[check_prime],
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=5000),
    ),
)
