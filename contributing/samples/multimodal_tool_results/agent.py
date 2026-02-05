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

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.genai import types

APP_NAME = "multimodal_tool_results"
USER_ID = "test_user"


def get_image():
  return [types.Part.from_uri(file_uri="gs://replace_with_your_image_uri")]


root_agent = LlmAgent(
    name="image_describing_agent",
    description="image describing agent",
    instruction="""Whatever the user says, get the image using the get_image tool, and describe it.""",
    model="gemini-2.0-flash",
    tools=[get_image],
)


app = App(
    name=APP_NAME,
    root_agent=root_agent,
    plugins=[MultimodalToolResultsPlugin()],
)
