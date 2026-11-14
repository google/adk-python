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

from unittest.mock import MagicMock

from google.adk.events.event_actions import EventActions
from google.adk.integrations.langchain import LangchainTool
from langchain_core.tools import Tool
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel
import pytest


@tool
async def async_add_with_annotation(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool
def sync_add_with_annotation(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool(return_direct=True)
def direct_add(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool(return_direct=True)
def direct_payload_with_error_key(x) -> dict:
  """Returns a payload that carries a falsy error key"""
  return {"error": None, "value": x}


async def async_add(x, y) -> int:
  return x + y


def sync_add(x, y) -> int:
  return x + y


def shout(text: str) -> str:
  return text.upper()


async def ashout(text: str) -> str:
  return text.upper()


def _declared_params(langchain_tool):
  """Returns the (names, required) parameters the model is shown."""
  declaration = langchain_tool._get_declaration()
  if declaration.parameters_json_schema is not None:
    schema = declaration.parameters_json_schema
    return list(schema.get("properties", {})), schema.get("required", [])
  schema = declaration.parameters
  return list(schema.properties or {}), schema.required or []


class AddSchema(BaseModel):
  x: int
  y: int


test_langchain_async_add_tool = StructuredTool.from_function(
    async_add,
    name="add",
    description="Adds two numbers",
    args_schema=AddSchema,
)

test_langchain_sync_add_tool = StructuredTool.from_function(
    sync_add,
    name="add",
    description="Adds two numbers",
    args_schema=AddSchema,
)


@pytest.mark.asyncio
async def test_raw_async_function_works():
  """Test that passing a raw async function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=test_langchain_async_add_tool)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_sync_function_works():
  """Test that passing a raw sync function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=test_langchain_sync_add_tool)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_async_function_with_annotation_works():
  """Test that passing a raw async function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=async_add_with_annotation)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_sync_function_with_annotation_works():
  """Test that passing a raw sync function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=sync_add_with_annotation)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_structured_tool_subclass_without_func_uses_run():
  """A StructuredTool subclass that overrides _run runs through _run."""

  class AddTool(StructuredTool):

    def _run(self, x: int, y: int) -> int:
      return x + y

  langchain_tool = LangchainTool(
      tool=AddTool(name="add", description="Adds", args_schema=AddSchema)
  )
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


def test_plain_structured_tool_without_func_raises():
  """A plain StructuredTool has no runnable function, so it is rejected."""
  tool = StructuredTool(name="add", description="Adds", args_schema=AddSchema)

  with pytest.raises(ValueError, match="must have a func"):
    LangchainTool(tool=tool)


def test_plain_tool_declares_func_params_not_config():
  """A plain Tool declares its func's parameters, not Tool._run's config."""
  langchain_tool = LangchainTool(
      tool=Tool(name="shout", func=shout, description="Shouts")
  )
  names, required = _declared_params(langchain_tool)
  assert names == ["text"]
  assert required == ["text"]
  assert "config" not in names


@pytest.mark.asyncio
async def test_plain_tool_run_async_calls_func():
  """A plain Tool runs its func with the model's arguments."""
  langchain_tool = LangchainTool(
      tool=Tool(name="shout", func=shout, description="Shouts")
  )
  result = await langchain_tool.run_async(
      args={"text": "hi"}, tool_context=MagicMock()
  )
  assert result == "HI"


@pytest.mark.asyncio
async def test_plain_tool_with_only_coroutine_works():
  """A plain Tool with only a coroutine declares and runs the coroutine."""
  langchain_tool = LangchainTool(
      tool=Tool(name="shout", func=None, coroutine=ashout, description="Shouts")
  )
  names, required = _declared_params(langchain_tool)
  assert names == ["text"]
  assert required == ["text"]
  result = await langchain_tool.run_async(
      args={"text": "hi"}, tool_context=MagicMock()
  )
  assert result == "HI"


@pytest.mark.asyncio
async def test_tool_subclass_overriding_run_uses_run():
  """A Tool subclass that overrides _run runs through _run."""

  class ShoutTool(Tool):

    def _run(self, text: str) -> str:
      return text.upper()

  langchain_tool = LangchainTool(
      tool=ShoutTool(name="shout", func=None, description="Shouts")
  )
  result = await langchain_tool.run_async(
      args={"text": "hi"}, tool_context=MagicMock()
  )
  assert result == "HI"


def test_plain_tool_without_func_or_coroutine_raises():
  """A plain Tool has no runnable function, so it is rejected."""
  tool = Tool(name="shout", func=None, description="Shouts")

  with pytest.raises(ValueError, match="must have a func"):
    LangchainTool(tool=tool)


@pytest.mark.asyncio
async def test_return_direct_sets_skip_summarization():
  """A tool with return_direct=True skips summarization on run."""
  langchain_tool = LangchainTool(tool=direct_add)
  assert langchain_tool._return_direct is True

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 2}, tool_context=tool_context
  )

  assert result == 3
  assert tool_context.actions.skip_summarization is True


@pytest.mark.asyncio
async def test_return_direct_leaves_skip_summarization_on_error():
  """A missing-argument error stays summarizable so the model can retry."""
  langchain_tool = LangchainTool(tool=direct_add)

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1}, tool_context=tool_context
  )

  assert "error" in result
  assert tool_context.actions.skip_summarization is None


@pytest.mark.asyncio
async def test_return_direct_skips_summarization_for_falsy_error_key():
  """A payload whose error key is falsy is a real result, not an error."""
  langchain_tool = LangchainTool(tool=direct_payload_with_error_key)

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1}, tool_context=tool_context
  )

  assert result == {"error": None, "value": 1}
  assert tool_context.actions.skip_summarization is True


@pytest.mark.asyncio
async def test_return_direct_default_false_leaves_skip_summarization():
  """A tool without return_direct does not touch skip_summarization."""
  langchain_tool = LangchainTool(tool=test_langchain_sync_add_tool)
  assert langchain_tool._return_direct is False

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=tool_context
  )

  assert result == 4
  assert tool_context.actions.skip_summarization is None
