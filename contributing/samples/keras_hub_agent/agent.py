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

import random

from google.adk.agents.llm_agent import Agent
from google.adk.models.keras_llm import KerasLlm


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result.

  Args:
      sides: The integer number of sides the die has.

  Returns:
      An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


async def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime.

  Args:
      nums: The list of numbers to check.

  Returns:
      A str indicating which number is prime.
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
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


# Agent using KerasHub for local model inference
keras_hub_agent = Agent(
    model=KerasLlm(
        model="gpt2_base_en",
        max_length=100,
        temperature=0.8,
        top_k=50,
    ),
    name="keras_hub_agent",
    description=(
        "A local AI assistant using KerasHub GPT-2 model that can roll dice and"
        " check prime numbers."
    ),
    instruction=(
        "You are a helpful local AI assistant powered by GPT-2. You can roll"
        " dice and check prime numbers. When asked to roll a die, call the"
        " roll_die tool with the number of sides. When checking prime numbers,"
        " call the check_prime tool with a list of integers."
    ),
    tools=[
        roll_die,
        check_prime,
    ],
)
