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

"""A voice agent: a text reasoner bracketed by STT and TTS models.

There is no ``live=`` switch and no live policy object: composition is the
``model``. `CascadeLive` wraps a plain text model in an STT transform and a
TTS transform. The tools and the instruction below are the same as in text
mode; callbacks, tool confirmation and long-running tools follow live-mode
semantics. This same agent still runs under ``run_async`` with no live
session at all.

`CASCADE_SPEECH_PROVIDER` picks the transforms: `gemini`, the default, uses
Gemini 3.5 Transcribe and Gemini 3.8 Flash-Lite TTS, and `elevenlabs` uses
ElevenLabs Scribe v2 Realtime and Flash v2.5. The reasoner thinks at its
lowest level, because thinking delays the first word of every reply by
seconds.
"""

from __future__ import annotations

import os

from google.adk import Agent
from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs import ElevenLabsTTS
from google.adk.live import CascadeLive
from google.adk.live import GeminiSTT
from google.adk.live import GeminiTTS
from google.adk.live import LiveEgress
from google.adk.live import LiveIngress
from google.genai import types


def _speech_transforms() -> tuple[LiveIngress, LiveEgress]:
  """Returns the STT and TTS of the provider `CASCADE_SPEECH_PROVIDER` names."""
  provider = (os.environ.get('CASCADE_SPEECH_PROVIDER') or 'gemini').lower()
  if provider == 'gemini':
    return GeminiSTT(), GeminiTTS(speech_config='Kore')
  if provider == 'elevenlabs':
    return ElevenLabsSTT(), ElevenLabsTTS()
  raise ValueError(
      "CASCADE_SPEECH_PROVIDER must be 'gemini' or 'elevenlabs', not"
      f' {provider!r}.'
  )


_stt, _tts = _speech_transforms()


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
        model='gemini-3.8-flash',
        stt=_stt,
        tts=_tts,
    ),
    name='cascade_agent',
    description=(
        'Voice agent that spells random words out and converts temperatures,'
        ' using Gemini or ElevenLabs for speech recognition and speech'
        ' synthesis.'
    ),
    instruction="""
      You are a voice assistant. Everything you say is spoken aloud by a
      speech synthesizer, so write for the ear, not for the page.

      Keep answers to one or two short sentences, except when spelling words
      out as described below. End every sentence with a full stop, a question
      mark or an exclamation mark: the synthesizer cuts your text at those
      boundaries, and a sentence that never ends waits before it is spoken.

      Never use markdown, bullet points, headings, code blocks or emoji.
      Write numbers, symbols and abbreviations the way you would say them out
      loud - "twenty three degrees", not "23 deg".

      When the user asks you to spell some random words, pick that many common
      words yourself and spell them out in a single reply. Announce each word
      as its own sentence, then spell it with each letter as its own short
      sentence. For example: "Apple. A. P. P. L. E." Do all of the words in
      one reply, one after another.

      When the user asks which words you spelled or what you just said, answer
      only from what you actually said earlier in this conversation. Do not
      pick new words, and do not spell anything again.

      When the user asks to convert a temperature, call the
      convert_temperature tool with the number and the unit it is currently
      in. Do not do the arithmetic yourself. Read the tool's answer back
      naturally rather than reciting it verbatim. Call the tool every time you
      are asked, even if the same question was already answered earlier.
    """,
    tools=[
        convert_temperature,
    ],
    generate_content_config=types.GenerateContentConfig(
        # The lowest level Gemini 3.8 Flash supports.
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW
        ),
    ),
)
