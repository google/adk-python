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

import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_response import LlmResponse
def create_inspector():
  """Callback to capture finish_reason."""
  captured = {"finish_reason": None}

  def inspector(
      callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse:
    captured["finish_reason"] = llm_response.finish_reason
    return llm_response

  inspector.captured = captured
  return inspector


# Create model with low max_tokens to trigger truncation
model = LiteLlm(
    model="gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_tokens=50,  # Intentionally low
)

inspector = create_inspector()

agent = Agent(
    model=model,
    name="test",
    instruction="Provide detailed explanations.",
    after_model_callback=inspector,
)
