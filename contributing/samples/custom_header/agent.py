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

"""
Examples showing how to add customized headers to model requests.
"""
from google.adk import Agent
from google.adk.models.llm_request import LlmRequest
from google.genai import types


async def add_additional_headers(callback_context, llm_request: LlmRequest):

  if not llm_request.config:
    llm_request.config = types.GenerateContentConfig()

  if not llm_request.config.http_options:
    llm_request.config.http_options = types.HttpOptions()
  if not llm_request.config.http_options.headers:
    llm_request.config.http_options.headers = {}
  llm_request.config.http_options.headers['customized-header'] = 'opaque-token'


root_agent = Agent(
    model='gemini-2.0-flash',
    name='hello_world_agent',
    description='hello world agent.',
    instruction="""You are a helpful assistant.
    """,
    before_model_callback=add_additional_headers,
)
