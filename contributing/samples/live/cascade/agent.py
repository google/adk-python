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

"""A voice agent: a text reasoner bracketed by ElevenLabs STT and TTS models.

There is no ``live=`` switch and no live policy object: composition is the
``model``. `CascadeLive` wraps a plain text model in an STT transform and a
TTS transform. The tools and the instruction below are the same as in text
mode; callbacks, tool confirmation and long-running tools follow live-mode
semantics. This same agent still runs under ``run_async`` with no live
session at all.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs import ElevenLabsTTS
from google.adk.live import CascadeLive


def spell_out(word: str) -> str:
  """Spell a word out one letter at a time.

  Args:
    word: The word to spell.

  Returns:
    A str with the letters separated, ready to be read aloud.
  """
  letters = [character.upper() for character in word if character.isalnum()]
  if not letters:
    return 'There are no letters in that to spell.'
  return f'{word} is spelled {", ".join(letters)}.'


def convert_temperature(value: float, from_unit: str) -> str:
  """Convert a temperature between Celsius and Fahrenheit.

  Args:
    value: The temperature to convert.
    from_unit: The unit the value is in, either Celsius or Fahrenheit.

  Returns:
    A str stating the converted temperature.
  """
  unit = from_unit.strip().lower()
  if unit in ('c', 'celsius'):
    converted = value * 9 / 5 + 32
    return f'{value:g} degrees Celsius is {converted:.1f} degrees Fahrenheit.'
  if unit in ('f', 'fahrenheit'):
    converted = (value - 32) * 5 / 9
    return f'{value:g} degrees Fahrenheit is {converted:.1f} degrees Celsius.'
  return f'I can convert Celsius and Fahrenheit, but not {from_unit}.'


root_agent = Agent(
    model=CascadeLive(
        # The reasoner: an ordinary text model, set as the `model`.
        model='gemini-3.5-flash',
        stt=ElevenLabsSTT(),
        tts=ElevenLabsTTS(),
    ),
    name='cascade_agent',
    description=(
        'Voice agent that spells words out and converts temperatures, using'
        ' ElevenLabs for speech recognition and speech synthesis.'
    ),
    instruction="""
      You are a voice assistant. Everything you say is spoken aloud by a
      speech synthesizer, so write for the ear, not for the page.

      Keep answers to one or two short sentences. End every sentence with a
      full stop, a question mark or an exclamation mark: the synthesizer cuts
      your text at those boundaries, and a sentence that never ends waits
      before it is spoken.

      Never use markdown, bullet points, headings, code blocks or emoji.
      Write numbers, symbols and abbreviations the way you would say them out
      loud - "twenty three degrees", not "23 deg".

      When the user asks how a word is spelled, call the spell_out tool. Do
      not spell it yourself.
      When the user asks to convert a temperature, call the
      convert_temperature tool with the number and the unit it is currently
      in. Do not do the arithmetic yourself.
      Read the tool's answer back naturally rather than reciting it verbatim.
      Call the tool every time you are asked, even if the same question was
      already answered earlier in this conversation. Never reuse an answer
      from an earlier turn.
    """,
    tools=[
        spell_out,
        convert_temperature,
    ],
)
