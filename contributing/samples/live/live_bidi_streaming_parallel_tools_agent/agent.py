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


from google.adk.agents.llm_agent import Agent


def turn_on_lights():
  """Turn on the lights."""
  print("turn_on_lights")
  return {"status": "OK"}


def turn_on_tv():
  """Turn on the tv."""
  print("turn_on_tv")
  return {"status": "OK"}


root_agent = Agent(
    # Find supported models in Vertex here: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
    model="gemini-live-2.5-flash-native-audio",  # Vertex
    # Find supported models in Gemini API here: https://ai.google.dev/gemini-api/docs/models
    # model='gemini-2.5-flash-native-audio-preview-12-2025',  # Gemini API
    name="Home_helper",
    instruction="Be polite and answer all user's questions.",
    tools=[turn_on_lights, turn_on_tv],
)
