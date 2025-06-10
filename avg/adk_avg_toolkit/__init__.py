"""
Agent Versatility Gear (AVG) Toolkit - Core Components
------------------------------------------------------

This package, `adk_avg_toolkit`, provides the fundamental building blocks
for creating sophisticated AI agents with a focus on modularity, configurability,
and deep expertise simulation.

Key components include:
- `KnowledgeConfigLoader`: For loading and accessing configurations for knowledge
  sources, ingestion pipelines, and agent abilities.
- `BaseExpertWrapper` (and example `ADKLlmExpertWrapper`): An abstract base class
  for wrapping underlying agent capabilities (e.g., Google ADK LlmAgents)
  to provide a standardized interface for interaction.
- `ExpertisePipelineManager`: For defining, managing, and executing complex
  workflows (pipelines) that orchestrate multiple expert wrappers to achieve
  advanced reasoning and task completion.

Custom error classes like `ConfigLoadError` and `ExpertisePipelineManagerError`
are also provided for robust error handling.
"""

from .knowledge_config_loader import KnowledgeConfigLoader, ConfigLoadError
from .base_expert_wrapper import BaseExpertWrapper, ADKLlmExpertWrapper
from .expertise_pipeline_manager import ExpertisePipelineManager, ExpertisePipelineManagerError

__all__ = [
    "KnowledgeConfigLoader",
    "ConfigLoadError",
    "BaseExpertWrapper",
    "ADKLlmExpertWrapper",
    "ExpertisePipelineManager",
    "ExpertisePipelineManagerError",
]
