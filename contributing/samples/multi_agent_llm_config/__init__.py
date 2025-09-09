import random

from google.adk.examples.example import Example
from google.adk.tools.example_tool import ExampleTool
from google.genai import types


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result."""
  return random.randint(1, sides)


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
