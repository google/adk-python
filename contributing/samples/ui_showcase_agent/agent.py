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

from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import get_user_choice_tool
from pydantic import BaseModel, Field
from google.genai import types # Assuming types.Part and types.ThinkingConfig are here

async def create_text_artifact(tool_context: ToolContext, content: str, filename: str = "showcase_text.txt"):
  """Creates a text artifact with the given content and filename.

  Args:
    tool_context: The context for tool execution.
    content: The text content to save in the artifact.
    filename: The name for the text artifact.
  """
  await tool_context.save_artifact(filename, types.Part(text=content))
  return f"Successfully created text artifact: {filename}"

async def create_image_artifact(tool_context: ToolContext, prompt: str, filename: str = "showcase_image_placeholder.png"):
  """Creates a placeholder image artifact.

  Args:
    tool_context: The context for tool execution.
    prompt: The prompt for the image (used in confirmation message).
    filename: The name for the image artifact.
  """
  # Simulate image generation by creating a text file with a .png extension
  # to see how the UI handles it.
  image_content = f"This is a placeholder for an image generated with prompt: '{prompt}'"
  await tool_context.save_artifact(filename, types.Part(text=image_content))
  # If direct bytes are preferred and types.Part supports it for non-images:
  # await tool_context.save_artifact(filename, types.Part.from_bytes(data=image_content.encode(), mime_type="image/png"))
  # For now, sticking to text part as per initial plan for simplicity.
  return f"Successfully created placeholder image artifact: {filename} (simulated for prompt: '{prompt}')"

async def greet_user(tool_context: ToolContext, name: str):
  """Greets the user with their name.

  Args:
    tool_context: The context for tool execution.
    name: The name of the user to greet.
  """
  return f"Hello, {name}! This demonstrates a simple function call."

class ComplexDataToolInput(BaseModel):
    text_input: str = Field(description="A piece of text information.")
    integer_input: int = Field(description="A numerical input.")
    boolean_input: bool = Field(description="A true/false flag.")
    list_input: list[str] = Field(description="A list of string items.")

async def process_complex_data(tool_context: ToolContext, data: ComplexDataToolInput):
  """Processes a complex data structure. 
  
  This tool is designed to showcase how the UI might render a form for
  complex input types when a user manually invokes a tool.
  Args:
    tool_context: The context for tool execution.
    data: A Pydantic model instance with text, integer, boolean, and list fields.
  """
  return (f"Processed complex data: Text='{data.text_input}', "
          f"Integer='{data.integer_input}', Boolean='{data.boolean_input}', "
          f"List='{data.list_input}'")

# Configure the planner to enable thinking display
ui_showcase_planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        # Optional: Adjust verbosity if needed, though default should be fine.
        # verbosity_level=types.ThinkingConfig.VerbosityLevel.FULL 
    )
)

ui_showcase_agent = Agent(
    model="gemini-2.0-flash-exp", # Using a model known from other samples
    name="ui_showcase_agent",
    description="A_SHOWCASE_AGENT_DESCRIPTION", # Placeholder, will be updated by a later step if needed / requested by user.
    instruction="""
You are a UI Showcase Agent. Your purpose is to demonstrate various ways information can be displayed in the agent interface.

To see different UI features, try the following:

1.  **See my thought process:** Ask a general question (e.g., "What is the capital of France?" or "Explain quantum physics in simple terms."). My thoughts will be displayed before I give an answer.
2.  **See a function call:** Ask me to greet someone (e.g., "Greet 'Jules'"). You will see the function call and its result.
3.  **Create a text artifact:** Ask me to create a text artifact (e.g., "Create a text artifact with content 'This is a test.' and filename 'my_test.txt'"). You can also just provide content, and a default filename will be used.
4.  **Create an image artifact (placeholder):** Ask me to create an image artifact (e.g., "Create an image artifact with prompt 'a sunny beach' and filename 'beach.png'"). This will create a placeholder text file with a .png extension to simulate image artifact display.
5.  **See user choice input:** Ask me to "request user choice with options Red, Green, Blue". I will then use a tool that should prompt you to select one of these options.
6.  **Observe complex tool input (manual invocation):** I have a tool called `process_complex_data` which expects several types of inputs (text, a number, a true/false value, and a list of items). If you try to run this tool manually from the ADK web interface (you might need to ask 'what are my tools?' to see it listed), the UI may present you with a form to fill in these details. This demonstrates how the system can generate input forms for tools.

I will use my tools when you ask for specific actions like creating artifacts or greeting.
For general questions, I will answer directly, showing my thinking process.
    """,
    tools=[
        create_text_artifact,
        create_image_artifact,
        greet_user,
        get_user_choice_tool,
        process_complex_data,
    ],
    planner=ui_showcase_planner,
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting( 
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE, # Adjusted for broad demonstration
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    ),
)

# Update the agent description dynamically based on its capabilities
ui_showcase_agent.description = (
    "Demonstrates UI features: thinking, function calls, artifacts, user choice prompts, and complex tool input forms (via manual tool invocation)."
)
