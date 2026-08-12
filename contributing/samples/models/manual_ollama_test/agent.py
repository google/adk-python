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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm

ollama_model = LiteLlm(model="ollama_chat/qwen2.5:7b")

hello_agent = LlmAgent(
    name="hello_step",
    instruction="Say hello to the user. Be concise.",
    model=ollama_model,
)

summarize_agent = LlmAgent(
    name="summarize_step",
    instruction="Summarize the previous assistant message in 5 words.",
    model=ollama_model,
)

root_agent = SequentialAgent(
    name="ollama_seq_test",
    description="Two-step sanity check for Ollama LiteLLM chat.",
    sub_agents=[hello_agent, summarize_agent],
)
