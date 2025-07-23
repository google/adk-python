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

"""
This agent aims to test the Langchain tool with Langchain's StructuredTool
"""
from google.adk.agents.llm_agent import Agent
from google.adk.tools.langchain_tool import LangchainTool
from langchain.tools import tool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel


async def add(x, y) -> int:
  return x + y


@tool
def minus(x, y) -> int:
  return x - y


class AddSchema(BaseModel):
  x: int
  y: int


class MinusSchema(BaseModel):
  x: int
  y: int


test_langchain_add_tool = StructuredTool.from_function(
    add,
    name="add",
    description="Adds two numbers",
    args_schema=AddSchema,
)

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="test_app",
    description="A helpful assistant for user questions.",
    instruction=(
        "You are a helpful assistant for user questions, you have access to a"
        " tool that adds two numbers."
    ),
    tools=[
        LangchainTool(tool=test_langchain_add_tool),
        LangchainTool(tool=minus),
    ],
)
