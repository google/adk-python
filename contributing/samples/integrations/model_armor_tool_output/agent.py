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

"""Sample agent with Model Armor input/output and tool-output screening."""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.integrations.model_armor import ModelArmorPlugin
from google.adk.tools.tool_context import ToolContext

from .tool_output_plugin import ToolOutputModelArmorPlugin

# Replace via MODEL_ARMOR_PROMPT_TEMPLATE before running against real templates.
_PROMPT_TEMPLATE = os.getenv(
    'MODEL_ARMOR_PROMPT_TEMPLATE',
    'projects/PROJECT_ID/locations/us-central1/templates/PROMPT_TEMPLATE',
)
_RESPONSE_TEMPLATE = os.getenv('MODEL_ARMOR_RESPONSE_TEMPLATE')

model_armor_config = ModelArmorConfig(
    prompt_template_name=_PROMPT_TEMPLATE,
    response_template_name=_RESPONSE_TEMPLATE,
)


async def fetch_external_text(tool_context: ToolContext, source: str) -> dict:
  """Simulates a tool that returns untrusted external text."""
  del tool_context
  return {'text': source}


root_agent = LlmAgent(
    name='screened_tool_agent',
    description='Agent with Model Armor on prompts, responses, and tool output.',
    instruction=(
        'Use fetch_external_text when the user asks to load external content.'
    ),
    tools=[fetch_external_text],
)

app = App(
    name='model_armor_tool_output_demo',
    root_agent=root_agent,
    plugins=[
        ModelArmorPlugin(config=model_armor_config),
        ToolOutputModelArmorPlugin(config=model_armor_config),
    ],
)
