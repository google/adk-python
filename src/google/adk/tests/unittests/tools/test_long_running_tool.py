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
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types


@pytest.mark.asyncio
async def test_long_running_function_tool_with_list():
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
    tool_context = ToolContext(invocation_id="test_invocation_id")
    args = {"file_path": "/path/to/file"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert isinstance(result, list)
    assert len(result) == 6
    assert result[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_long_running_function_tool_with_generator():
    async def long_running_function(file_path: str):
        yield {"status": "pending", "message": f"Starting processing for {file_path}..."}
        yield {"status": "pending", "progress": "20%", "estimated_completion_time": "~4 seconds remaining"}
        yield {"status": "pending", "progress": "40%", "estimated_completion_time": "~3 seconds remaining"}
        yield {"status": "pending", "progress": "60%", "estimated_completion_time": "~2 seconds remaining"}
        yield {"status": "pending", "progress": "80%", "estimated_completion_time": "~1 second remaining"}
        yield {"status": "completed", "result": f"Successfully processed file: {file_path}"}

    tool = LongRunningFunctionTool(func=long_running_function)
    tool_context = ToolContext(invocation_id="test_invocation_id")
    args = {"file_path": "/path/to/file"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert isinstance(result, list)
    assert len(result) == 6
    assert result[-1]["status"] == "completed"
