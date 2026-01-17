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

"""Tests for ToolCodeGenerator."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from google.genai import types

from google.adk.code_executors.tool_code_generator import generate_full_code_with_stubs
from google.adk.code_executors.tool_code_generator import generate_runtime_header
from google.adk.code_executors.tool_code_generator import generate_system_prompt
from google.adk.code_executors.tool_code_generator import generate_tool_stubs
from google.adk.tools.base_tool import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        params: dict = None,
    ):
        super().__init__(name=name, description=description)
        self._params = params or {}

    def _get_declaration(self):
        properties = {}
        required = []

        for param_name, param_info in self._params.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required", False):
                required.append(param_name)

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="object",
                properties=properties,
                required=required if required else None,
            ),
        )

    async def run_async(self, *, args, tool_context):
        return {"result": "mock"}


class TestGenerateRuntimeHeader:
    """Tests for generate_runtime_header function."""

    def test_generates_valid_header(self):
        """Test that the header contains required elements."""
        url = "http://localhost:8765"
        header = generate_runtime_header(url)

        # Should contain the URL
        assert url in header

        # Should contain helper functions
        assert "_call_adk_tool" in header
        assert "final_answer" in header
        assert "__get_tool_traces" in header

        # Should be valid Python syntax
        compile(header, "<string>", "exec")

    def test_header_with_different_urls(self):
        """Test header generation with different URLs."""
        urls = [
            "http://localhost:8765",
            "http://host.docker.internal:9999",
            "http://172.17.0.1:8765",
        ]

        for url in urls:
            header = generate_runtime_header(url)
            assert url in header

    def test_header_contains_trace_collection(self):
        """Test that header contains trace collection code."""
        header = generate_runtime_header("http://localhost:8765")

        assert "__ADK_TOOL_TRACES" in header
        assert "__get_tool_traces" in header
        assert "__clear_tool_traces" in header

    def test_header_contains_final_answer_marker(self):
        """Test that header contains final answer marker."""
        header = generate_runtime_header("http://localhost:8765")

        assert "__FINAL_ANSWER__" in header
        assert "final_answer" in header


class TestGenerateToolStubs:
    """Tests for generate_tool_stubs function."""

    def test_generates_stub_for_tool(self):
        """Test generating stub for a single tool."""
        tool = MockTool(
            name="search",
            description="Search for information",
            params={
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True,
                }
            },
        )

        stubs = generate_tool_stubs([tool])

        # Should contain function definition
        assert "def search(" in stubs
        assert "query" in stubs

        # Should be valid Python
        compile(stubs, "<string>", "exec")

    def test_generates_stubs_for_multiple_tools(self):
        """Test generating stubs for multiple tools."""
        tools = [
            MockTool(name="tool1", description="First tool"),
            MockTool(name="tool2", description="Second tool"),
            MockTool(name="tool3", description="Third tool"),
        ]

        stubs = generate_tool_stubs(tools)

        assert "def tool1(" in stubs
        assert "def tool2(" in stubs
        assert "def tool3(" in stubs

    def test_stub_includes_docstring(self):
        """Test that stubs include docstrings."""
        tool = MockTool(
            name="my_tool",
            description="A tool that does something useful",
        )

        stubs = generate_tool_stubs([tool])

        assert '"""' in stubs
        assert "A tool that does something useful" in stubs

    def test_stub_includes_type_hints(self):
        """Test that stubs include type hints."""
        tool = MockTool(
            name="typed_tool",
            description="A typed tool",
            params={
                "count": {"type": "integer", "description": "A count"},
                "name": {"type": "string", "description": "A name"},
                "enabled": {"type": "boolean", "description": "Is enabled"},
            },
        )

        stubs = generate_tool_stubs([tool])

        assert "int" in stubs
        assert "str" in stubs
        assert "bool" in stubs

    def test_empty_tool_list(self):
        """Test generating stubs for empty tool list."""
        stubs = generate_tool_stubs([])

        # Should still be valid Python
        compile(stubs, "<string>", "exec")


class TestGenerateSystemPrompt:
    """Tests for generate_system_prompt function."""

    def test_generates_prompt_with_tools(self):
        """Test generating system prompt with tools."""
        tools = [
            MockTool(
                name="search",
                description="Search the web",
                params={"query": {"type": "string", "required": True}},
            ),
        ]

        prompt = generate_system_prompt(tools)

        # Should contain tool documentation
        assert "search" in prompt
        assert "Search the web" in prompt

        # Should contain usage instructions
        assert "tool_code" in prompt
        assert "final_answer" in prompt

    def test_generates_prompt_with_custom_instruction(self):
        """Test generating prompt with custom instruction."""
        tools = []
        custom = "Always be polite and helpful."

        prompt = generate_system_prompt(tools, custom_instruction=custom)

        assert custom in prompt

    def test_generates_prompt_with_examples(self):
        """Test that prompt contains examples."""
        tools = []
        prompt = generate_system_prompt(tools)

        assert "Example" in prompt
        assert "```tool_code" in prompt

    def test_generates_prompt_with_parameter_docs(self):
        """Test that prompt includes parameter documentation."""
        tools = [
            MockTool(
                name="get_weather",
                description="Get weather for a city",
                params={
                    "city": {
                        "type": "string",
                        "description": "The city name",
                        "required": True,
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "required": False,
                    },
                },
            ),
        ]

        prompt = generate_system_prompt(tools)

        assert "city" in prompt
        assert "units" in prompt
        assert "required" in prompt.lower() or "optional" in prompt.lower()


class TestGenerateFullCodeWithStubs:
    """Tests for generate_full_code_with_stubs function."""

    def test_generates_complete_code(self):
        """Test generating complete executable code."""
        tools = [MockTool(name="my_tool", description="A tool")]
        user_code = "result = my_tool()\nprint(result)"

        full_code = generate_full_code_with_stubs(
            user_code=user_code,
            tools=tools,
            tool_server_url="http://localhost:8765",
        )

        # Should contain runtime header
        assert "_call_adk_tool" in full_code

        # Should contain tool stub
        assert "def my_tool(" in full_code

        # Should contain user code
        assert user_code in full_code

        # Should be valid Python
        compile(full_code, "<string>", "exec")

    def test_generated_code_outputs_traces(self):
        """Test that generated code outputs traces."""
        tools = []
        user_code = "x = 1"

        full_code = generate_full_code_with_stubs(
            user_code=user_code,
            tools=tools,
            tool_server_url="http://localhost:8765",
        )

        assert "__TOOL_TRACE__" in full_code

    def test_generated_code_is_executable(self):
        """Test that generated code can be compiled."""
        tools = [
            MockTool(name="tool_a", description="Tool A"),
            MockTool(name="tool_b", description="Tool B"),
        ]
        user_code = """
result_a = tool_a()
result_b = tool_b()
print(result_a, result_b)
"""

        full_code = generate_full_code_with_stubs(
            user_code=user_code,
            tools=tools,
            tool_server_url="http://localhost:8765",
        )

        # Should compile without errors
        compile(full_code, "<string>", "exec")
