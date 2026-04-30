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

from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openrouter/openai/gpt-5.2')


root_agent = Agent(
    name='openrouter_agent',
    model=LiteLlm(
        model=OPENROUTER_MODEL,
        api_key=os.getenv('OPENROUTER_API_KEY'),
        api_base=OPENROUTER_API_BASE,
    ),
    description='A simple ADK agent that uses OpenRouter through LiteLLM.',
    instruction=(
        'You are a concise assistant running through OpenRouter. Answer the'
        ' user directly and mention when a question needs current or external'
        ' information.'
    ),
)
