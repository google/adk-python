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

"""Skills module for ADK.

Skills bundle related tools with orchestration logic for efficient
programmatic tool calling (PTC). This module provides:

Core abstractions:
- BaseSkill: Abstract base class for all skills
- SkillConfig: Configuration for skill execution
- SkillsManager: Skill registration and execution management

Agent Skills standard support (https://agentskills.io):
- MarkdownSkill: Load skills from SKILL.md files
- MarkdownSkillMetadata: Parsed frontmatter metadata
- AgentSkillLoader: Discover and load skills from directories

Execution:
- ScriptExecutor: Execute bundled scripts safely
- ScriptExecutionResult: Result from script execution

Tool integration:
- SkillTool: Wrap skills as ADK tools for LLM invocation
- create_skill_tools: Convenience function to create tool wrappers

Built-in skills are bundled in this module:
- bqml: BigQuery ML skill for training and inference
- bq-ai-operator: BigQuery AI operations (text generation, embeddings)

Example usage:
  ```python
  from google.adk.skills import AgentSkillLoader, SkillTool, SkillsManager

  # Load Agent Skills standard skills
  loader = AgentSkillLoader()
  loader.add_skill_directory("./skills")

  # Register with manager
  manager = SkillsManager()
  loader.register_all(manager)

  # Create tools for LLM
  skill_tools = [SkillTool(s) for s in manager.get_all_skills()]

  # Generate discovery prompt
  prompt = loader.generate_discovery_prompt()
  ```
"""

from .agent_skill_loader import AgentSkillLoader
# Core abstractions
from .base_skill import BaseSkill
from .base_skill import SkillConfig
# Agent Skills standard support
from .markdown_skill import MarkdownSkill
from .markdown_skill import MarkdownSkillMetadata
# Execution
from .script_executor import ScriptExecutionError
from .script_executor import ScriptExecutionResult
from .script_executor import ScriptExecutor
from .skill_manager import SkillInvocationResult
from .skill_manager import SkillsManager
# Tool integration
from .skill_tool import create_skill_tools
from .skill_tool import SkillTool

# Skills directory path for loading bundled skills
from pathlib import Path
SKILLS_DIR = Path(__file__).parent

__all__ = [
    # Skills directory
    "SKILLS_DIR",
    # Core abstractions
    "BaseSkill",
    "SkillConfig",
    "SkillInvocationResult",
    "SkillsManager",
    # Agent Skills standard support
    "MarkdownSkill",
    "MarkdownSkillMetadata",
    "AgentSkillLoader",
    # Execution
    "ScriptExecutor",
    "ScriptExecutionResult",
    "ScriptExecutionError",
    # Tool integration
    "SkillTool",
    "create_skill_tools",
]
