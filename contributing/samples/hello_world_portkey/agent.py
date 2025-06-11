"""Hello World Portkey sample mirroring the LiteLLM dice example.

Run with:
   adk run contributing/samples/hello_world_portkey

Requires:
   pip install portkey-ai
   export PORTKEY_API_KEY=pk_live_...
"""

import random

from google.adk.agents import Agent
from google.adk.models import PortkeyLlm


# ---------------------------------------------------------------------------
# Simple tool: return a random dad-joke from a short list
# ---------------------------------------------------------------------------


_JOKES = [
    "Why did the developer go broke? Because they used up all their cache!",
    "Why do functions always break up? Because they have constant arguments.",
    "Debugging: Removing the needles from the haystack.",
]


def get_random_joke() -> str:  # noqa: D401
  """Return a random joke string."""

  return random.choice(_JOKES)


# ---------------------------------------------------------------------------
# Root agent definition
# ---------------------------------------------------------------------------


root_agent = Agent(
    name="hello_portkey",
    model=PortkeyLlm(model="@openai-9d6e0b/gpt-4o"),  # relies on PORTKEY_API_KEY env var
    description="Demo agent that can tell programming jokes.",
    instruction=(
        "You are a friendly assistant.  When the user asks for a joke you MUST "
        "call the get_random_joke function.  For all other questions answer "
        "directly."
    ),
    tools=[get_random_joke],
) 