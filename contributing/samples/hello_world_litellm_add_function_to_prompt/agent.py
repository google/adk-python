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

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from langchain_core.utils.function_calling import convert_to_openai_function


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def check_prime(number: int) -> str:
  """Check if a given number is prime.

  Args:
    number: The input number to check.

  Returns:
    A str indicating the number is prime or not.
  """
  if number <= 1:
    return f"{number} is not prime."
  is_prime = True
  for i in range(2, int(number**0.5) + 1):
    if number % i == 0:
      is_prime = False
      break
  if is_prime:
    return f"{number} is prime."
  else:
    return f"{number} is not prime."


root_agent = Agent(
    model=LiteLlm(
        model="vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
        # If the model is not trained with functions and you would like to
        # enable function calling, you can add functions to the models, and the
        # functions will be added to the prompts during inferences.
        functions=[
            convert_to_openai_function(roll_die),
            convert_to_openai_function(check_prime),
        ],
    ),
    name="data_processing_agent",
    description="""You are a helpful assistant.""",
    instruction="""
      You are a helpful assistant, and call tools optionally.
      If call tools, the tool format should be in json, and the tool arguments should be parsed from users inputs.
    """,
    tools=[
        roll_die,
        check_prime,
    ],
)
