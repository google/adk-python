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

"""Sample agent demonstrating Synap memory tools.

Synap (https://maximem.ai) is a managed long-term memory layer for AI agents.
The `maximem-synap-google-adk` package exposes Synap as ADK FunctionTools so
the agent can search and store memories across sessions.

Setup:
    pip install maximem-synap-google-adk maximem-synap
    export SYNAP_API_KEY=<your-key>  # https://synap.maximem.ai

Open source integration package:
https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations/synap-google-adk
"""

import os

from google.adk.agents.llm_agent import Agent
from maximem_synap import MaximemSynapSDK
from synap_google_adk import create_synap_tools

# Initialize the Synap SDK once at module load.
sdk = MaximemSynapSDK(api_key=os.environ["SYNAP_API_KEY"])

# `create_synap_tools` returns [search_memory, store_memory] FunctionTool instances.
synap_tools = create_synap_tools(
    sdk=sdk,
    user_id="demo-user-001",
    customer_id="demo-customer",
)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="memory_assistant",
    instruction=(
        "You are a helpful assistant with long-term memory. "
        "Use search_memory to recall what you know about the user. "
        "Use store_memory to save important new facts the user mentions."
    ),
    tools=synap_tools,
)
