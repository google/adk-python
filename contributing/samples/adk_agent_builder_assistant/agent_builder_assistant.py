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

"""Agent factory for creating Agent Builder Assistant with embedded schema."""

from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Union

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import BaseLlm
from google.adk.tools import AgentTool
from google.adk.tools import FunctionTool
from google.genai import types

from .sub_agents.google_search_agent import create_google_search_agent
from .sub_agents.url_context_agent import create_url_context_agent
from .tools.cleanup_unused_files import cleanup_unused_files
from .tools.delete_files import delete_files
from .tools.explore_project import explore_project
from .tools.read_config_files import read_config_files
from .tools.read_files import read_files
from .tools.search_adk_knowledge import search_adk_knowledge
from .tools.search_adk_source import search_adk_source
from .tools.write_config_files import write_config_files
from .tools.write_files import write_files
from .utils import load_agent_config_schema


class AgentBuilderAssistant:
  """Agent Builder Assistant factory for creating configured instances."""

  @staticmethod
  def create_agent(
      model: Union[str, BaseLlm] = "gemini-2.5-flash",
      working_directory: Optional[str] = None,
  ) -> LlmAgent:
    """Create Agent Builder Assistant with embedded ADK AgentConfig schema.

    Args:
      model: Model to use for the assistant (default: gemini-2.5-flash)
      working_directory: Working directory for path resolution (default: current
        working directory)

    Returns:
      Configured LlmAgent with embedded ADK AgentConfig schema
    """
    # Load full ADK AgentConfig schema directly into instruction context
    instruction = AgentBuilderAssistant._load_instruction_with_schema(model)

    # TOOL ARCHITECTURE: Hybrid approach using both AgentTools and FunctionTools
    #
    # Why use sub-agents for built-in tools?
    # - ADK's built-in tools (google_search, url_context) are designed as agents
    # - AgentTool wrapper allows integrating them into our agent's tool collection
    # - Maintains compatibility with existing ADK tool ecosystem

    # Built-in ADK tools wrapped as sub-agents
    google_search_agent = create_google_search_agent()
    url_context_agent = create_url_context_agent()
    agent_tools = [
        AgentTool(google_search_agent),
        AgentTool(url_context_agent),
    ]

    # CUSTOM FUNCTION TOOLS: Agent Builder specific capabilities
    #
    # Why FunctionTool pattern?
    # - Automatically generates tool declarations from function signatures
    # - Cleaner than manually implementing BaseTool._get_declaration()
    # - Type hints and docstrings become tool descriptions automatically

    # Core agent building tools
    custom_tools = [
        FunctionTool(read_config_files),  # Read/parse multiple YAML configs
        FunctionTool(
            write_config_files
        ),  # Write/validate multiple YAML configs
        FunctionTool(explore_project),  # Analyze project structure
        # File management tools (multi-file support)
        FunctionTool(read_files),  # Read multiple files
        FunctionTool(write_files),  # Write multiple files
        FunctionTool(delete_files),  # Delete multiple files
        FunctionTool(cleanup_unused_files),
        # ADK source code search (regex-based)
        FunctionTool(search_adk_source),  # Search ADK source with regex
        # ADK knowledge search
        FunctionTool(search_adk_knowledge),  # Search ADK knowledge base
    ]

    # Combine all tools
    all_tools = agent_tools + custom_tools

    # Create agent directly using LlmAgent constructor
    agent = LlmAgent(
        name="agent_builder_assistant",
        description=(
            "Intelligent assistant for building ADK multi-agent systems "
            "using YAML configurations"
        ),
        instruction=instruction,
        model=model,
        tools=all_tools,
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=8192,
        ),
    )

    return agent

  @staticmethod
  def _load_schema() -> str:
    """Load ADK AgentConfig.json schema content and format for YAML embedding."""

    # CENTRALIZED ADK AGENTCONFIG SCHEMA LOADING: Use common utility function
    # This avoids duplication across multiple files and provides consistent
    # ADK AgentConfig schema loading with caching and error handling.
    schema_content = load_agent_config_schema(
        raw_format=True,  # Get as JSON string
    )

    # Format as indented code block for instruction embedding
    #
    # Why indentation is needed:
    # - The ADK AgentConfig schema gets embedded into instruction templates using .format()
    # - Proper indentation maintains readability in the final instruction
    # - Code block markers (```) help LLMs recognize this as structured data
    #
    # Example final instruction format:
    #   "Here is the ADK AgentConfig schema:
    #   ```json
    #     {"type": "object", "properties": {...}}
    #   ```"
    lines = schema_content.split("\n")
    indented_lines = ["  " + line for line in lines]  # 2-space indent
    return "```json\n" + "\n".join(indented_lines) + "\n  ```"

  @staticmethod
  def _load_instruction_with_schema(
      model: Union[str, BaseLlm],
  ) -> Callable[[ReadonlyContext], str]:
    """Load instruction template and embed ADK AgentConfig schema content."""
    instruction_template = (
        AgentBuilderAssistant._load_embedded_schema_instruction_template()
    )
    schema_content = AgentBuilderAssistant._load_schema()

    # Get model string for template replacement
    model_str = (
        str(model)
        if isinstance(model, str)
        else getattr(model, "model_name", str(model))
    )

    # Return a function that accepts ReadonlyContext and returns the instruction
    def instruction_provider(context: ReadonlyContext) -> str:
      # Extract project folder name from session state
      project_folder_name = AgentBuilderAssistant._extract_project_folder_name(
          context
      )

      # Fill the instruction template with all variables
      instruction_text = instruction_template.format(
          schema_content=schema_content,
          default_model=model_str,
          project_folder_name=project_folder_name,
      )
      return instruction_text

    return instruction_provider

  @staticmethod
  def _extract_project_folder_name(context: ReadonlyContext) -> str:
    """Extract project folder name from session state using resolve_file_path."""
    from .utils.resolve_root_directory import resolve_file_path

    session_state = context._invocation_context.session.state

    # Use resolve_file_path to get the full resolved path for "."
    # This handles all the root_directory resolution logic consistently
    resolved_path = resolve_file_path(".", session_state)

    # Extract the project folder name from the resolved path
    project_folder_name = resolved_path.name

    # Fallback to "project" if we somehow get an empty name
    if not project_folder_name:
      project_folder_name = "project"

    return project_folder_name

  @staticmethod
  def _load_embedded_schema_instruction_template() -> str:
    """Load instruction template for embedded ADK AgentConfig schema mode."""
    template_path = Path(__file__).parent / "instruction_embedded.template"

    if not template_path.exists():
      raise FileNotFoundError(
          f"Instruction template not found at {template_path}"
      )

    with open(template_path, "r", encoding="utf-8") as f:
      return f.read()
