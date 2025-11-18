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
import textwrap
from typing import Any
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

  _CORE_SCHEMA_DEF_NAMES: tuple[str, ...] = (
      "LlmAgentConfig",
      "LoopAgentConfig",
      "ParallelAgentConfig",
      "SequentialAgentConfig",
      "BaseAgentConfig",
      "AgentRefConfig",
      "CodeConfig",
      "ArgumentConfig",
      "ToolArgsConfig",
      "google__adk__tools__tool_configs__ToolConfig",
  )
  _GEN_CONFIG_FIELDS: tuple[str, ...] = (
      "temperature",
      "topP",
      "topK",
      "maxOutputTokens",
  )

  @staticmethod
  def create_agent(
      model: Union[str, BaseLlm] = "gemini-2.5-pro",
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

    schema_dict = load_agent_config_schema(raw_format=False)
    subset = AgentBuilderAssistant._extract_core_schema(schema_dict)
    return AgentBuilderAssistant._build_schema_reference(subset)

  @staticmethod
  def _build_schema_reference(schema: dict[str, Any]) -> str:
    """Create compact AgentConfig reference text for prompt embedding."""

    defs: dict[str, Any] = schema.get("$defs", {})
    top_level_fields: dict[str, Any] = schema.get("properties", {})
    wrapper = textwrap.TextWrapper(width=78)
    lines: list[str] = []

    def add(text: str = "", indent: int = 0) -> None:
      """Append wrapped text with indentation."""
      if not text:
        lines.append("")
        return
      indent_str = " " * indent
      wrapper.initial_indent = indent_str
      wrapper.subsequent_indent = indent_str
      lines.extend(wrapper.fill(text).split("\n"))

    add("ADK AgentConfig quick reference")
    add("--------------------------------")

    add()
    add("LlmAgent (agent_class: LlmAgent)")
    add(
        "Required fields: name, instruction. ADK best practice is to always set"
        " model explicitly.",
        indent=2,
    )
    add("Optional fields:", indent=2)
    add("agent_class: defaults to LlmAgent; keep for clarity.", indent=4)
    add("description: short summary string.", indent=4)
    add("sub_agents: list of AgentRef entries (see below).", indent=4)
    add(
        "before_agent_callbacks / after_agent_callbacks: list of CodeConfig "
        "entries that run before or after the agent loop.",
        indent=4,
    )
    add("model: string model id (required in practice).", indent=4)
    add(
        "disallow_transfer_to_parent / disallow_transfer_to_peers: booleans to "
        "restrict automatic transfer.",
        indent=4,
    )
    add(
        "input_schema / output_schema: JSON schema objects to validate inputs "
        "and outputs.",
        indent=4,
    )
    add("output_key: name to store agent output in session context.", indent=4)
    add(
        "include_contents: bool; include tool/LLM contents in response.",
        indent=4,
    )
    add("tools: list of ToolConfig entries (see below).", indent=4)
    add(
        "before_model_callbacks / after_model_callbacks: list of CodeConfig "
        "entries around LLM calls.",
        indent=4,
    )
    add(
        "before_tool_callbacks / after_tool_callbacks: list of CodeConfig "
        "entries around tool calls.",
        indent=4,
    )
    add(
        "generate_content_config: passes directly to google.genai "
        "GenerateContentConfig (supporting temperature, topP, topK, "
        "maxOutputTokens, safetySettings, responseSchema, routingConfig,"
        " etc.).",
        indent=4,
    )

    add()
    add("Workflow agents (LoopAgent, ParallelAgent, SequentialAgent)")
    add(
        "Share BaseAgent fields: agent_class, name, description, sub_agents, "
        "before/after_agent_callbacks. Never declare model, instruction, or "
        "tools on workflow orchestrators.",
        indent=2,
    )
    add(
        "LoopAgent adds max_iterations (int) controlling iteration cap.",
        indent=2,
    )

    add()
    add("AgentRef")
    add(
        "Used inside sub_agents lists. Provide either config_path (string path "
        "to another YAML file) or code (dotted Python reference) to locate the "
        "sub-agent definition.",
        indent=2,
    )

    add()
    add("ToolConfig")
    add(
        "Items inside tools arrays. Required field name (string). For built-in "
        "tools use the exported short name, for custom tools use the dotted "
        "module path.",
        indent=2,
    )
    add(
        "args: optional object of additional keyword arguments. Use simple "
        "key-value pairs (ToolArgsConfig) or structured ArgumentConfig entries "
        "when a list is required by callbacks.",
        indent=2,
    )

    add()
    add("ArgumentConfig")
    add(
        "Represents a single argument. value is required and may be any JSON "
        "type. name is optional (null allowed). Often used in callback args.",
        indent=2,
    )

    add()
    add("CodeConfig")
    add(
        "References Python code for callbacks or dynamic tool creation."
        " Requires name (dotted path). args is an optional list of"
        " ArgumentConfig items executed when invoking the function.",
        indent=2,
    )

    add()
    add("GenerateContentConfig highlights")
    add(
        "Controls LLM generation behavior. Common fields: maxOutputTokens, "
        "temperature, topP, topK, candidateCount, responseMimeType, "
        "responseSchema/responseJsonSchema, automaticFunctionCalling, "
        "safetySettings, routingConfig; see Vertex AI GenAI docs for full "
        "semantics.",
        indent=2,
    )

    add()
    add(
        "All other schema definitions in AgentConfig.json remain available but "
        "are rarely needed for typical agent setups. Refer to the source file "
        "for exhaustive field descriptions when implementing advanced configs.",
    )

    if top_level_fields:
      add()
      add("Top-level AgentConfig fields (from schema)")
      for field_name in sorted(top_level_fields):
        description = top_level_fields[field_name].get("description", "")
        if description:
          add(f"{field_name}: {description}", indent=2)
        else:
          add(field_name, indent=2)

    if defs:
      add()
      add("Additional schema definitions")
      for def_name in sorted(defs):
        description = defs[def_name].get("description", "")
        if description:
          add(f"{def_name}: {description}", indent=2)
        else:
          add(def_name, indent=2)

    return "```text\n" + "\n".join(lines) + "\n```"

  @staticmethod
  def _extract_core_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return only the schema nodes surfaced by the assistant."""

    defs = schema.get("$defs", {})
    filtered_defs: dict[str, Any] = {}
    for key in AgentBuilderAssistant._CORE_SCHEMA_DEF_NAMES:
      if key in defs:
        filtered_defs[key] = defs[key]

    gen_config = defs.get("GenerateContentConfig")
    if gen_config:
      properties = gen_config.get("properties", {})
      filtered_defs["GenerateContentConfig"] = {
          "title": gen_config.get("title", "GenerateContentConfig"),
          "description": (
              "Common LLM generation knobs exposed by the Agent Builder."
          ),
          "type": "object",
          "additionalProperties": False,
          "properties": {
              key: properties[key]
              for key in AgentBuilderAssistant._GEN_CONFIG_FIELDS
              if key in properties
          },
      }

    return {
        "$defs": filtered_defs,
        "properties": schema.get("properties", {}),
    }

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
