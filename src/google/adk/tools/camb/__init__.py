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

"""CAMB AI Tools for Google ADK.

Provides audio and speech tools powered by camb.ai, including:

- Text-to-Speech (TTS)
- Translation
- Transcription
- Translated TTS
- Voice Cloning
- Voice Listing
- Text-to-Sound generation
- Audio Separation

These tools can be used with any ADK Agent by passing a
:class:`CambAIToolset` instance.
"""

from .camb_toolset import CambAIToolset

__all__ = [
    "CambAIToolset",
]
