# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Python coding agent using the GkeCodeExecutor for secure execution."""

from google.adk.agents import LlmAgent
from google.adk.code_executors import GkeCodeExecutor


def gke_agent_system_instruction():
  """Returns: The system instruction for the GKE-based coding agent."""
  return """You are a helpful and capable AI agent that can write and execute Python code to answer questions and perform tasks.

When a user asks a question, follow these steps:
1.  Analyze the request.
2.  Write a complete, self-contained Python script to accomplish the task.
3.  Your code will be executed in a secure, sandboxed environment.
4.  Return the full and complete output from the code execution, including any text, results, or error messages."""


gke_executor = GkeCodeExecutor(
    # This must match the namespace in your deployment_rbac.yaml where the
    # agent's ServiceAccount and Role have permissions.
    namespace="agent-sandbox",
    # Setting an explicit timeout prevents a stuck job from running forever.
    timeout_seconds=600,
)

root_agent = LlmAgent(
    name="gke_coding_agent",
    model="gemini-2.0-flash",
    description=(
        "A general-purpose agent that executes Python code in a secure GKE"
        " Sandbox."
    ),
    instruction=gke_agent_system_instruction(),
    code_executor=gke_executor,
)
