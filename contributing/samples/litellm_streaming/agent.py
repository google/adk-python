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

"""LiteLLM sample agent for SSE text streaming."""

from __future__ import annotations

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name='litellm_streaming_agent',
    model=LiteLlm(model='gemini/gemini-2.5-flash'),
    description='A LiteLLM agent used for streaming text responses.',
    instruction='You are a verbose assistant',
)
