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


from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool


def update_current_time(callback_context: CallbackContext):
  callback_context.state['_time'] = datetime.now().isoformat()


root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='memory_agent',
    description='agent that have access to memory tools.',
    before_agent_callback=update_current_time,
    instruction="""\
You are an agent that help user answer questions.

Current time: {_time}
""",
    tools=[
        load_memory_tool,
        preload_memory_tool,
    ],
)
