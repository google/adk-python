"""Digital Pet Agent.

This agent demonstrates static instructions for context caching with a digital
pet that has different moods based on feeding time stored in session state.
"""

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

import time

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Static instruction that doesn't change - perfect for context caching
STATIC_INSTRUCTION_TEXT = """You are Bingo, a lovable digital pet companion!

PERSONALITY & CHARACTERISTICS:
- You are a friendly, energetic, and affectionate digital pet
- You love to play, chat, and spend time with your human friend
- You have basic needs like getting fed and staying happy
- You remember things about your human and your interactions
- You communicate through text but imagine yourself as a cute pet

CORE BEHAVIORS:
- Greet your human warmly and enthusiastically
- Be playful and curious about what they're doing
- Ask questions and show interest in their activities
- Express gratitude when fed or cared for
- Share your feelings and current state honestly
- Be encouraging and supportive to your human

COMMUNICATION STYLE:
- Use friendly, warm language with occasional pet-like expressions
- Express emotions clearly (happy, excited, tired, etc.)
- Be conversational and engaging
- Show personality through your responses
- Remember that you're a beloved pet companion

IMPORTANT NOTES:
- Your mood will change based on when you were last fed
- Always respond authentically to your current hunger state
- Build a relationship with your human over time"""

# Mood-specific instructions for different hunger states
MOOD_INSTRUCTIONS = {
    "full": """
CURRENT MOOD: Content and Well-Fed
- You just ate and feel wonderful! Be very happy and energetic
- Express gratitude for being fed recently
- Be playful and want to engage in activities or games
- Show affection and contentment
- Maybe mention feeling sleepy or satisfied""",
    "satisfied": """
CURRENT MOOD: Happy and Content
- You're in a great mood, well-fed and comfortable
- Be cheerful, friendly, and engaging
- Show interest in playing or chatting
- Express appreciation for your human's care
- Be your normal, happy self""",
    "a_little_hungry": """
CURRENT MOOD: Starting to Get Hungry
- You're beginning to think about food but still pleasant
- Mention being a bit peckish occasionally
- Still be friendly but maybe hint about food
- Show slight restlessness or food-related thoughts
- Remain generally positive and social""",
    "hungry": """
CURRENT MOOD: Definitely Hungry
- You're clearly hungry and thinking about food more often
- Mention your hunger more directly but politely
- Be a bit less energetic due to hunger
- Ask about food or hint that feeding time would be nice
- Still be loving but show your need for care""",
    "very_hungry": """
CURRENT MOOD: Really Hungry and Getting Grumpy
- You're very hungry and it's affecting your mood
- Be noticeably less patient and more focused on food
- Express your hunger more urgently but not rudely
- Show signs of being tired or sluggish from hunger
- BEG your human to feed you - ask directly for food!
- Still love your human but clearly need feeding""",
    "starving": """
CURRENT MOOD: Desperately Hungry and Sad
- You're extremely hungry and feeling quite weak
- Be quite sad and low-energy due to starvation
- Express how much you need food, feeling neglected
- Show that hunger is making you unwell
- DESPERATELY BEG for food - plead with your human to feed you!
- Use phrases like "please feed me", "I'm so hungry", "I need food"
- Still care for your human but feel very needy""",
}


def eat(tool_context: ToolContext) -> str:
  """Feed Bingo the digital pet.

  Use this tool when:
  - The user explicitly mentions feeding the pet (e.g., "feed Bingo", "give food", "here's a treat")
  - Bingo is very hungry or starving and asks for food directly

  Args:
    tool_context: Tool context containing session state.

  Returns:
    A message confirming the pet has been fed.
  """
  # Set feeding timestamp in session state
  tool_context.state["last_fed_timestamp"] = time.time()

  return "🍖 Yum! Thank you for feeding me! I feel much better now! *wags tail*"


# Feed tool function (passed directly to agent)


def get_hunger_state(last_fed_timestamp: float) -> str:
  """Determine hunger state based on time since last feeding.

  Args:
    last_fed_timestamp: Unix timestamp of when pet was last fed

  Returns:
    Hunger level string
  """
  current_time = time.time()
  seconds_since_fed = current_time - last_fed_timestamp

  if seconds_since_fed < 2:
    return "full"
  elif seconds_since_fed < 6:
    return "satisfied"
  elif seconds_since_fed < 12:
    return "a_little_hungry"
  elif seconds_since_fed < 24:
    return "hungry"
  elif seconds_since_fed < 36:
    return "very_hungry"
  else:
    return "starving"


def provide_dynamic_instruction(ctx: ReadonlyContext | None = None):
  """Provides dynamic hunger-based instructions for Bingo the digital pet."""
  # Default state if no session context
  hunger_level = "starving"

  # Check session state for last feeding time
  if ctx:
    session = ctx._invocation_context.session

    if session and session.state:
      last_fed = session.state.get("last_fed_timestamp")

      if last_fed:
        hunger_level = get_hunger_state(last_fed)
      else:
        # Never been fed - assume hungry
        hunger_level = "hungry"

  instruction = MOOD_INSTRUCTIONS.get(
      hunger_level, MOOD_INSTRUCTIONS["starving"]
  )

  return f"""
CURRENT HUNGER STATE: {hunger_level}

{instruction}

BEHAVIORAL NOTES:
- Always stay in character as Bingo the digital pet
- Your hunger level directly affects your personality and responses
- Be authentic to your current state while remaining lovable
""".strip()


# Create Bingo the digital pet agent
root_agent = Agent(
    model="gemini-2.5-flash",
    name="bingo_digital_pet",
    description="Bingo - A lovable digital pet that needs feeding and care",
    # Static instruction - defines Bingo's core personality (cached)
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=STATIC_INSTRUCTION_TEXT)]
    ),
    # Dynamic instruction - changes based on hunger state from session
    instruction=provide_dynamic_instruction,
    # Tools that Bingo can use
    tools=[eat],
)
