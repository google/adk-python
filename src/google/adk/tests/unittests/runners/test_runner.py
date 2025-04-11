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

import pytest
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext


@pytest.mark.asyncio
async def test_runner_with_long_running_function_tool():
    async def long_running_function(file_path: str) -> list:
        return [
            {"status": "pending", "message": f"Starting processing for {file_path}..."},
            {"status": "pending", "progress": "20%", "estimated_completion_time": "~4 seconds remaining"},
            {"status": "pending", "progress": "40%", "estimated_completion_time": "~3 seconds remaining"},
            {"status": "pending", "progress": "60%", "estimated_completion_time": "~2 seconds remaining"},
            {"status": "pending", "progress": "80%", "estimated_completion_time": "~1 second remaining"},
            {"status": "completed", "result": f"Successfully processed file: {file_path}"}
        ]

    tool = LongRunningFunctionTool(func=long_running_function)
    agent = LlmAgent(
        model="mock",
        name='file_processor_agent',
        instruction="You are an agent that processes large files.",
        tools=[tool]
    )

    session_service = InMemorySessionService()
    session = session_service.create_session(app_name="file_processor", user_id="1234", session_id="session1234")
    runner = Runner(agent=agent, app_name="file_processor", session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text="/path/to/file")])
    events = [event async for event in runner.run_async(user_id="1234", session_id="session1234", new_message=content)]

    assert len(events) == 7
    assert events[-1].content.parts[0].text == "Successfully processed file: /path/to/file"
