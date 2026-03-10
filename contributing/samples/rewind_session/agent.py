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

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


async def update_state(tool_context: ToolContext, key: str, value: str) -> dict:
  """Updates a state value."""
  tool_context.state[key] = value
  return {"status": f"Updated state '{key}' to '{value}'"}


async def load_state(tool_context: ToolContext, key: str) -> dict:
  """Loads a state value."""
  return {key: tool_context.state.get(key)}


async def save_artifact(
    tool_context: ToolContext, filename: str, content: str
) -> dict:
  """Saves an artifact with the given filename and content."""
  artifact_bytes = content.encode("utf-8")
  artifact_part = types.Part(
      inline_data=types.Blob(mime_type="text/plain", data=artifact_bytes)
  )
  version = await tool_context.save_artifact(filename, artifact_part)
  return {"status": "success", "filename": filename, "version": version}


async def load_artifact(tool_context: ToolContext, filename: str) -> dict:
  """Loads an artifact with the given filename."""
  artifact = await tool_context.load_artifact(filename)
  if not artifact:
    return {"error": f"Artifact '{filename}' not found"}
  content = artifact.inline_data.data.decode("utf-8")
  return {"filename": filename, "content": content}


# Create the agent
root_agent = Agent(
    name="state_agent",
    model="gemini-2.0-flash",
    instruction="""You are an agent that manages state and artifacts.

    You can:
    - Update state value
    - Load state value
    - Save artifact
    - Load artifact

    Use the appropriate tool based on what the user asks for.""",
    tools=[
        update_state,
        load_state,
        save_artifact,
        load_artifact,
    ],
)
