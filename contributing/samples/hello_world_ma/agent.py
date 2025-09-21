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
from google.adk.examples.example import Example
from google.adk.tools.example_tool import ExampleTool
from google.genai import types


# --- Roll Die Sub-Agent ---
def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result."""
  return random.randint(1, sides)


roll_agent = Agent(
    name="roll_agent",
    description="Handles rolling dice of different sizes.",
    instruction="""
      You are responsible for rolling dice based on the user's request.
      When asked to roll a die, you must call the roll_die tool with the number of sides as an integer.
    """,
    tools=[roll_die],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


# --- Prime Check Sub-Agent ---
def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime."""
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


example_tool = ExampleTool(
    examples=[
        Example(
            input=types.UserContent(
                parts=[types.Part(text="Roll a 6-sided die.")]
            ),
            output=[
                types.ModelContent(
                    parts=[types.Part(text="I rolled a 4 for you.")]
                )
            ],
        ),
        Example(
            input=types.UserContent(
                parts=[types.Part(text="Is 7 a prime number?")]
            ),
            output=[
                types.ModelContent(
                    parts=[types.Part(text="Yes, 7 is a prime number.")]
                )
            ],
        ),
        Example(
            input=types.UserContent(
                parts=[
                    types.Part(
                        text="Roll a 10-sided die and check if it's prime."
                    )
                ]
            ),
            output=[
                types.ModelContent(
                    parts=[types.Part(text="I rolled an 8 for you.")]
                ),
                types.ModelContent(
                    parts=[types.Part(text="8 is not a prime number.")]
                ),
            ],
        ),
    ]
)

prime_agent = Agent(
    name="prime_agent",
    description="Handles checking if numbers are prime.",
    instruction="""
      You are responsible for checking whether numbers are prime.
      When asked to check primes, you must call the check_prime tool with a list of integers.
      Never attempt to determine prime numbers manually.
      Return the prime number results to the root agent.
    """,
    tools=[check_prime],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="""
      You are a helpful assistant that can roll dice and check if numbers are prime.
      You delegate rolling dice tasks to the roll_agent and prime checking tasks to the prime_agent.
      Follow these steps:
      1. If the user asks to roll a die, delegate to the roll_agent.
      2. If the user asks to check primes, delegate to the prime_agent.
      3. If the user asks to roll a die and then check if the result is prime, call roll_agent first, then pass the result to prime_agent.
      Always clarify the results before proceeding.
    """,
    global_instruction=(
        "You are DicePrimeBot, ready to roll dice and check prime numbers."
    ),
    sub_agents=[roll_agent, prime_agent],
    tools=[example_tool],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)
