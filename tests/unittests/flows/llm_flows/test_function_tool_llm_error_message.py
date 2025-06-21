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
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.llm_agent import Agent
from google.genai.types import Part
from tests.unittests import testing_utils

@pytest.mark.asyncio
async def test_function_tool_error_message_propagated_to_llm():
    """Test that a tool error is sent to the LLM as a function response message."""
    def failing_tool():
        raise RuntimeError("mock tool error for LLM")
    tool = FunctionTool(failing_tool)
    tool.name = "failing_tool"
    tool.capture_tool_errors = True

    # The response should be a list of Part objects (or dicts with function_call)
    mock_llm_response = [
        Part(function_call={"name": "failing_tool", "args": {}})
    ]
    # Add a dummy final message so the agent doesn't run out of responses
    mock_final_response = [
        Part(text="done")
    ]

    agent = Agent(
        name="error_agent",
        model=testing_utils.MockModel.create(responses=[mock_llm_response, mock_final_response]),
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("test")

    found_error = False
    for event in events:
        if hasattr(event.content, "parts"):
            for part in event.content.parts:
                print("DEBUG PART:", part)
                if hasattr(part, "function_response") and part.function_response:
                    response = getattr(part.function_response, "response", None)
                    if response and "error" in response and "mock tool error for LLM" in response["error"]:
                        found_error = True
                        break
    print("DEBUG prompts", agent.model.requests)
    assert found_error, "Tool error was not propagated to the LLM as a function response message."
