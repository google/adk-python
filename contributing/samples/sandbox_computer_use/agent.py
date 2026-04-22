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

"""Sample agent using Vertex AI Agent Engine Sandbox for computer use.

This sample demonstrates how to use the AgentEngineSandboxComputer with ADK
to create a computer use agent that operates in a remote sandbox environment.

Prerequisites:
  1. A GCP project with Agent Engine setup (https://docs.cloud.google.com/agent-builder/agent-engine/set-up)
  2. A service account with roles/iam.serviceAccountTokenCreator permission
  3. Environment variables in contributing/samples/.env:
     - GOOGLE_CLOUD_PROJECT: Your GCP project ID
     - VMAAS_SERVICE_ACCOUNT: Your service account email
     - VMAAS_SANDBOX_NAME: (Optional) Existing sandbox resource name for BYOS mode

Usage:
  # Run via ADK web UI
  adk web contributing/samples/sandbox_computer_use

  # Run via main.py
  cd contributing/samples
  python -m sandbox_computer_use.main
"""

import os

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.integrations.vmaas import AgentEngineSandboxComputer
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration from environment variables
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
SERVICE_ACCOUNT = os.environ.get("VMAAS_SERVICE_ACCOUNT")

# Optional: Use existing sandbox (BYOS mode)
# Format: projects/{project}/locations/{location}/reasoningEngines/{id}/sandboxEnvironments/{id}
SANDBOX_NAME = os.environ.get("SANDBOX_NAME") or os.environ.get(
    "VMAAS_SANDBOX_NAME"
)

# Create the sandbox computer
sandbox_computer = AgentEngineSandboxComputer(
    project_id=PROJECT_ID,
    service_account_email=SERVICE_ACCOUNT,
    sandbox_name=SANDBOX_NAME,
    search_engine_url="https://www.google.com",
)

# Create agent with the computer use toolset
root_agent = Agent(
    model="gemini-2.5-computer-use-preview-10-2025",
    name="sandbox_computer_use_agent",
    description=(
        "A computer use agent that operates a browser in a remote Vertex AI"
        " sandbox environment to complete user tasks."
    ),
    instruction="""You are a computer use agent that can operate a web browser
to help users complete tasks. You have access to browser controls including:
- Navigation (go to URLs, back, forward, search)
- Mouse actions (click, hover, scroll, drag and drop)
- Keyboard input (type text, key combinations)
- Screenshots (to see the current state)

When given a task:
1. Think about what steps are needed to accomplish it
2. Take actions one at a time, observing the results
3. If something doesn't work, try alternative approaches
4. Report back when the task is complete or if you encounter issues

Be careful with sensitive information and always respect website terms of service.
""",
    tools=[ComputerUseToolset(computer=sandbox_computer)],
)
