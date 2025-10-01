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

from .sub_agents.google_search_agent import create_google_search_agent
from .sub_agents.url_context_agent import create_url_context_agent
from .tools.cleanup_unused_files import cleanup_unused_files
from .tools.delete_files import delete_files
from .tools.explore_project import explore_project
from .tools.read_config_files import read_config_files
from .tools.read_files import read_files
from .tools.resolve_root_directory import resolve_root_directory
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
    instruction = AgentBuilderAssistant._load_instruction_with_schema(
        model, working_directory
    )

    # TOOL ARCHITECTURE: Hybrid approach using both AgentTools and FunctionTools
    #
    # Why use sub-agents for built-in tools?
    # - ADK's built-in tools (google_search, url_context) are designed as agents
    # - AgentTool wrapper allows integrating them into our agent's tool collection
    # - Maintains compatibility with existing ADK tool ecosystem

    # Built-in ADK tools wrapped as sub-agents
    google_search_agent = create_google_search_agent()
    url_context_agent = create_url_context_agent()
    agent_tools = [AgentTool(google_search_agent), AgentTool(url_context_agent)]

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
        # Working directory context tools
        FunctionTool(resolve_root_directory),
        # File management tools (multi-file support)
        FunctionTool(read_files),  # Read multiple files
        FunctionTool(write_files),  # Write multiple files
        FunctionTool(delete_files),  # Delete multiple files
        FunctionTool(cleanup_unused_files),
        # ADK source code search (regex-based)
        FunctionTool(search_adk_source),  # Search ADK source with regex
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
        escape_braces=True,  # Escape braces for template embedding
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
      working_directory: Optional[str] = None,
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

    # Fill the instruction template with ADK AgentConfig schema content and default model
    instruction_text = instruction_template.format(
        schema_content=schema_content, default_model=model_str
    )

    # Return a function that accepts ReadonlyContext and returns the instruction
    def instruction_provider(context: ReadonlyContext) -> str:
      return AgentBuilderAssistant._compile_instruction_with_context(
          instruction_text, context, working_directory
      )

    return instruction_provider

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

  @staticmethod
  def _compile_instruction_with_context(
      instruction_text: str,
      context: ReadonlyContext,
      working_directory: Optional[str] = None,
  ) -> str:
    """Compile instruction with session context and working directory information.

    This method enhances instructions with:
    1. Working directory information for path resolution
    2. Session-based root directory binding if available

    Args:
      instruction_text: Base instruction text
      context: ReadonlyContext from the agent session
      working_directory: Optional working directory for path resolution

    Returns:
      Enhanced instruction text with context information
    """
    import os

    # Get working directory (use provided or current working directory)
    actual_working_dir = working_directory or os.getcwd()

    # Check for existing root directory in session state
    session_root_directory = context._invocation_context.session.state.get(
        "root_directory"
    )

    # Compile additional context information
    context_info = f"""

## SESSION CONTEXT

**Working Directory**: `{actual_working_dir}`
- Use this as the base directory for path resolution when calling resolve_root_directory
- Pass this as the working_directory parameter to resolve_root_directory tool

"""

    if session_root_directory:
      context_info += f"""**Established Root Directory**: `{session_root_directory}`
- This session is bound to root directory: {session_root_directory}
- DO NOT ask the user for root directory - use this established path
- All agent building should happen within this root directory
- If user wants to work in a different directory, ask them to start a new chat session

"""
    else:
      context_info += f"""**Root Directory**: Not yet established
- You MUST ask the user for their desired root directory first
- Use resolve_root_directory tool to validate the path
- Once confirmed, this session will be bound to that root directory

"""

    context_info += """**Session Binding Rules**:
- Each chat session is bound to ONE root directory
- Once established, work only within that root directory
- To switch directories, user must start a new chat session
- Always verify paths using resolve_root_directory tool before creating files

"""

    return instruction_text + context_info
