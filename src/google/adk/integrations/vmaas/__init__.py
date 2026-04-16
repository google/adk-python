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

"""Vertex AI Agent Engine Computer Use Sandbox integration.

This module provides a BaseComputer implementation that uses Vertex AI
Agent Engine Computer Use Sandbox as the remote browser environment.

Example:
  ```python
  from google.adk.integrations.vmaas import AgentEngineSandboxComputer
  from google.adk.tools.computer_use import ComputerUseToolset

  computer = AgentEngineSandboxComputer(
      project_id="my-project",
      service_account_email="sa@my-project.iam.gserviceaccount.com",
  )
  toolset = ComputerUseToolset(computer=computer)
  agent = Agent(tools=[toolset], ...)
  ```
"""

from .sandbox_client import SandboxClient
from .sandbox_computer import AgentEngineSandboxComputer

__all__ = [
    "AgentEngineSandboxComputer",
    "SandboxClient",
]
