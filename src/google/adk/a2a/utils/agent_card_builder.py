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

from __future__ import annotations

import logging
from typing import Dict
from typing import List

from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentProvider
from a2a.types import AgentSkill
from a2a.types import SecurityScheme

from .. import _compat
from ...agents.base_agent import BaseAgent
from ...agents.llm_agent import LlmAgent
from ...agents.loop_agent import LoopAgent
from ...agents.parallel_agent import ParallelAgent
from ...agents.sequential_agent import SequentialAgent
from ...tools.example_tool import ExampleTool
from ...workflow import BaseNode
from ...workflow import START
from ...workflow import Workflow
from ..experimental import a2a_experimental

logger = logging.getLogger('google_adk.' + __name__)


@a2a_experimental
class AgentCardBuilder:
  """Builder class for creating agent cards from ADK agents.

  This class provides functionality to convert ADK agents into A2A agent cards,
  including extracting skills, capabilities, and metadata from various agent
  types.
  """

  def __init__(
      self,
      *,
      agent: BaseAgent | Workflow,
      rpc_url: str | None = None,
      capabilities: AgentCapabilities | None = None,
      doc_url: str | None = None,
      provider: AgentProvider | None = None,
      agent_version: str | None = None,
      security_schemes: Dict[str, SecurityScheme] | None = None,
  ):
    if not agent:
      raise ValueError('Agent cannot be None or empty.')
    if not isinstance(agent, (BaseAgent, Workflow)):
      raise TypeError(
          'AgentCardBuilder requires a BaseAgent or Workflow, got '
          f'{type(agent).__name__}.'
      )

    self._agent = agent
    self._rpc_url = rpc_url or 'http://localhost:80/a2a'
    self._capabilities = capabilities or AgentCapabilities()
    self._doc_url = doc_url
    self._provider = provider
    self._security_schemes = security_schemes
    self._agent_version = agent_version or '0.0.1'

  async def build(self) -> AgentCard:
    """Build and return the complete agent card."""
    try:
      primary_skills = await _build_primary_skills(self._agent)
      sub_agent_skills = await _build_sub_agent_skills(self._agent)
      all_skills = primary_skills + sub_agent_skills

      return _compat.build_agent_card(
          name=self._agent.name,
          description=self._agent.description or 'An ADK Agent',
          version=self._agent_version,
          url=self._rpc_url,
          protocol_binding=getattr(
              _compat.TP_JSONRPC, 'value', _compat.TP_JSONRPC
          ),
          skills=all_skills,
          capabilities=self._capabilities,
          provider=self._provider,
          security_schemes=self._security_schemes,
          doc_url=self._doc_url,
          default_input_modes=['text/plain'],
          default_output_modes=['text/plain'],
          supports_authenticated_extended_card=False,
      )
    except Exception as e:
      raise RuntimeError(
          f'Failed to build agent card for {self._agent.name}: {e}'
      ) from e


# Module-level helper functions
def _iter_child_nodes(agent: BaseNode) -> List[BaseNode]:
  """Returns the immediate child nodes of an agent or a workflow."""
  if isinstance(agent, BaseAgent):
    return list(agent.sub_agents)
  if isinstance(agent, Workflow) and agent.graph is not None:
    return [n for n in agent.graph.nodes if n.name != START.name]
  return []


async def _build_primary_skills(agent: BaseNode) -> List[AgentSkill]:
  """Build skills for any node type."""
  if isinstance(agent, LlmAgent):
    return await _build_llm_agent_skills(agent)
  else:
    return await _build_non_llm_agent_skills(agent)


async def _build_llm_agent_skills(agent: LlmAgent) -> List[AgentSkill]:
  """Build skills for LLM agent."""
  skills = []

  # 1. Agent skill (main model skill). The card is a discovery document served
  # without authentication, so the description comes from the agent's own
  # public description and never from its instructions.
  agent_description = _build_agent_description(agent)
  agent_examples = await _extract_examples_from_agent(agent)

  skills.append(
      AgentSkill(
          id=agent.name,
          name='model',
          description=agent_description,
          examples=_extract_inputs_from_examples(agent_examples),
          input_modes=_get_input_modes(agent),
          output_modes=_get_output_modes(agent),
          tags=['llm'],
      )
  )

  # 2. Tool skills
  if agent.tools:
    tool_skills = await _build_tool_skills(agent)
    skills.extend(tool_skills)

  # 3. Planner skill
  if agent.planner:
    skills.append(_build_planner_skill(agent))

  # 4. Code executor skill
  if agent.code_executor:
    skills.append(_build_code_executor_skill(agent))

  return skills


async def _build_sub_agent_skills(agent: BaseNode) -> List[AgentSkill]:
  """Build skills for all child nodes (sub-agents or workflow nodes)."""
  sub_agent_skills = []
  for sub_agent in _iter_child_nodes(agent):
    try:
      sub_skills = await _build_primary_skills(sub_agent)
      for skill in sub_skills:
        # Create a new skill instance to avoid modifying original if shared
        aggregated_skill = AgentSkill(
            id=f'{sub_agent.name}_{skill.id}',
            name=f'{sub_agent.name}: {skill.name}',
            description=skill.description,
            examples=skill.examples,
            input_modes=skill.input_modes,
            output_modes=skill.output_modes,
            tags=[f'sub_agent:{sub_agent.name}'] + list(skill.tags or []),
        )
        sub_agent_skills.append(aggregated_skill)
    except Exception as e:
      # Log warning but continue with other sub-agents
      logger.warning(
          'Failed to build skills for sub-agent %s: %s', sub_agent.name, e
      )
      continue

  return sub_agent_skills


async def _build_tool_skills(agent: LlmAgent) -> List[AgentSkill]:
  """Build skills for agent tools."""
  tool_skills = []
  canonical_tools = await agent.canonical_tools()

  for tool in canonical_tools:
    # Skip example tools as they're handled separately
    if isinstance(tool, ExampleTool):
      continue

    tool_name = (
        tool.name
        if hasattr(tool, 'name') and tool.name
        else tool.__class__.__name__
    )

    tool_skills.append(
        AgentSkill(
            id=f'{agent.name}-{tool_name}',
            name=tool_name,
            description=getattr(tool, 'description', f'Tool: {tool_name}'),
            examples=None,
            input_modes=None,
            output_modes=None,
            tags=['llm', 'tools'],
        )
    )

  return tool_skills


def _build_planner_skill(agent: LlmAgent) -> AgentSkill:
  """Build planner skill for LLM agent."""
  return AgentSkill(
      id=f'{agent.name}-planner',
      name='planning',
      description='Can think about the tasks to do and make plans',
      examples=None,
      input_modes=None,
      output_modes=None,
      tags=['llm', 'planning'],
  )


def _build_code_executor_skill(agent: LlmAgent) -> AgentSkill:
  """Build code executor skill for LLM agent."""
  return AgentSkill(
      id=f'{agent.name}-code-executor',
      name='code-execution',
      description='Can execute code',
      examples=None,
      input_modes=None,
      output_modes=None,
      tags=['llm', 'code_execution'],
  )


async def _build_non_llm_agent_skills(agent: BaseNode) -> List[AgentSkill]:
  """Build skills for non-LLM agents and workflow nodes."""
  skills = []

  # 1. Agent skill (main agent skill)
  agent_description = _build_agent_description(agent)
  agent_examples = await _extract_examples_from_agent(agent)

  # Determine agent type and name
  agent_type = _get_agent_type(agent)
  agent_name = _get_agent_skill_name(agent)

  skills.append(
      AgentSkill(
          id=agent.name,
          name=agent_name,
          description=agent_description,
          examples=_extract_inputs_from_examples(agent_examples),
          input_modes=_get_input_modes(agent),
          output_modes=_get_output_modes(agent),
          tags=[agent_type],
      )
  )

  # 2. Orchestration skill (for agents/workflows with child nodes)
  if _iter_child_nodes(agent):
    orchestration_skill = _build_orchestration_skill(agent, agent_type)
    if orchestration_skill:
      skills.append(orchestration_skill)

  return skills


def _build_orchestration_skill(
    agent: BaseNode, agent_type: str
) -> AgentSkill | None:
  """Build orchestration skill for agents/workflows with child nodes."""
  sub_agent_descriptions = []
  for sub_agent in _iter_child_nodes(agent):
    description = sub_agent.description or 'No description'
    sub_agent_descriptions.append(f'{sub_agent.name}: {description}')

  if not sub_agent_descriptions:
    return None

  return AgentSkill(
      id=f'{agent.name}-sub-agents',
      name='sub-agents',
      description='Orchestrates: ' + '; '.join(sub_agent_descriptions),
      examples=None,
      input_modes=None,
      output_modes=None,
      tags=[agent_type, 'orchestration'],
  )


def _get_agent_type(agent: BaseNode) -> str:
  """Get the agent type for tagging."""
  if isinstance(agent, LlmAgent):
    return 'llm'
  elif isinstance(agent, SequentialAgent):
    return 'sequential_workflow'
  elif isinstance(agent, ParallelAgent):
    return 'parallel_workflow'
  elif isinstance(agent, LoopAgent):
    return 'loop_workflow'
  elif isinstance(agent, Workflow):
    return 'graph_workflow'
  else:
    return 'custom_agent'


def _get_agent_skill_name(agent: BaseNode) -> str:
  """Get the skill name based on agent type."""
  if isinstance(agent, LlmAgent):
    return 'model'
  elif isinstance(agent, (SequentialAgent, ParallelAgent, LoopAgent, Workflow)):
    return 'workflow'
  else:
    return 'custom'


def _build_agent_description(agent: BaseNode) -> str:
  """Build agent description from agent.description and workflow-specific descriptions."""
  description_parts = []

  # Add agent description
  if agent.description:
    description_parts.append(agent.description)

  # Add workflow-specific descriptions for non-LLM agents
  if not isinstance(agent, LlmAgent):
    workflow_description = _get_workflow_description(agent)
    if workflow_description:
      description_parts.append(workflow_description)

  return (
      ' '.join(description_parts)
      if description_parts
      else _get_default_description(agent)
  )


def _get_workflow_description(agent: BaseNode) -> str | None:
  """Get workflow-specific description for non-LLM agents and workflows."""
  if not _iter_child_nodes(agent):
    return None

  if isinstance(agent, SequentialAgent):
    return _build_sequential_description(agent)
  elif isinstance(agent, ParallelAgent):
    return _build_parallel_description(agent)
  elif isinstance(agent, LoopAgent):
    return _build_loop_description(agent)
  elif isinstance(agent, Workflow):
    return _build_graph_workflow_description(agent)

  return None


def _build_sequential_description(agent: SequentialAgent) -> str:
  """Build description for sequential workflow agent."""
  descriptions = []
  for i, sub_agent in enumerate(agent.sub_agents, 1):
    sub_description = (
        sub_agent.description or f'execute the {sub_agent.name} agent'
    )
    if i == 1:
      descriptions.append(f'First, this agent will {sub_description}')
    elif i == len(agent.sub_agents):
      descriptions.append(f'Finally, this agent will {sub_description}')
    else:
      descriptions.append(f'Then, this agent will {sub_description}')
  return ' '.join(descriptions) + '.'


def _build_parallel_description(agent: ParallelAgent) -> str:
  """Build description for parallel workflow agent."""
  descriptions = []
  for i, sub_agent in enumerate(agent.sub_agents):
    sub_description = (
        sub_agent.description or f'execute the {sub_agent.name} agent'
    )
    if i == 0:
      descriptions.append(f'This agent will {sub_description}')
    elif i == len(agent.sub_agents) - 1:
      descriptions.append(f'and {sub_description}')
    else:
      descriptions.append(f', {sub_description}')
  return ' '.join(descriptions) + ' simultaneously.'


def _build_loop_description(agent: LoopAgent) -> str:
  """Build description for loop workflow agent."""
  max_iterations = (
      'unlimited' if agent.max_iterations is None else agent.max_iterations
  )
  descriptions = []
  for i, sub_agent in enumerate(agent.sub_agents):
    sub_description = (
        sub_agent.description or f'execute the {sub_agent.name} agent'
    )
    if i == 0:
      descriptions.append(f'This agent will {sub_description}')
    elif i == len(agent.sub_agents) - 1:
      descriptions.append(f'and {sub_description}')
    else:
      descriptions.append(f', {sub_description}')
  return (
      f"{' '.join(descriptions)} in a loop (max {max_iterations} iterations)."
  )


def _build_graph_workflow_description(workflow: Workflow) -> str:
  """Build description for a graph-based Workflow."""
  child_nodes = _iter_child_nodes(workflow)
  descriptions = []
  for node in child_nodes:
    node_description = (
        node.description.rstrip('.')
        if node.description
        else f'execute the {node.name} node'
    )
    descriptions.append(f'{node.name}: {node_description}')
  return (
      'This workflow orchestrates the following nodes: '
      + '; '.join(descriptions)
      + '.'
  )


def _get_default_description(agent: BaseNode) -> str:
  """Get default description based on agent type."""
  agent_type_descriptions = {
      LlmAgent: 'An LLM-based agent',
      SequentialAgent: 'A sequential workflow agent',
      ParallelAgent: 'A parallel workflow agent',
      LoopAgent: 'A loop workflow agent',
      Workflow: 'A graph-based workflow agent',
  }

  for agent_type, description in agent_type_descriptions.items():
    if isinstance(agent, agent_type):
      return description

  return 'A custom agent'


def _extract_inputs_from_examples(
    examples: list[dict[str, object]] | None,
) -> list[str]:
  """Extracts only the input strings so they can be added to an AgentSkill."""
  if examples is None:
    return []

  extracted_inputs: list[str] = []
  for example in examples:
    example_input = example.get('input')
    if not isinstance(example_input, dict):
      continue

    parts = example_input.get('parts')
    if isinstance(parts, list):
      part_texts: list[str] = []
      for part in parts:
        if not isinstance(part, dict):
          continue
        text = part.get('text')
        if isinstance(text, str):
          part_texts.append(text)
      if part_texts:
        extracted_inputs.append('\n'.join(part_texts))
    else:
      text = example_input.get('text')
      if isinstance(text, str):
        extracted_inputs.append(text)

  return extracted_inputs


async def _extract_examples_from_agent(
    agent: BaseNode,
) -> list[dict[str, object]] | None:
  """Extract examples from example_tool if configured, otherwise none."""
  if not isinstance(agent, LlmAgent):
    return None

  # First, try to find example_tool in tools
  try:
    canonical_tools = await agent.canonical_tools()
    for tool in canonical_tools:
      if isinstance(tool, ExampleTool):
        examples = _convert_example_tool_examples(tool)
        if examples is not None:
          return examples
  except Exception as e:
    logger.warning('Failed to extract examples from tools: %s', e)

  # Examples come only from a declared example_tool, never mined out of the
  # instruction, which is not publishable content.
  return None


def _serialize_example_content(content: object) -> object:
  model_dump = getattr(content, 'model_dump', None)
  if callable(model_dump):
    serialized: object = model_dump()
    return serialized
  return content


def _convert_example_tool_examples(
    tool: ExampleTool,
) -> list[dict[str, object]] | None:
  """Convert ExampleTool examples to the expected format."""
  if not isinstance(tool.examples, list):
    logger.debug(
        'Skipping dynamic ExampleTool provider when building an agent card'
    )
    return None

  examples: list[dict[str, object]] = []
  for example in tool.examples:
    examples.append({
        'input': _serialize_example_content(example.input),
        'output': [
            _serialize_example_content(output) for output in example.output
        ],
    })
  return examples


def _get_input_modes(agent: BaseNode) -> List[str] | None:
  """Get input modes based on agent model."""
  if not isinstance(agent, LlmAgent):
    return None

  # This could be enhanced to check model capabilities
  # For now, return None to use default_input_modes
  return None


def _get_output_modes(agent: BaseNode) -> List[str] | None:
  """Get output modes from Agent.generate_content_config.response_modalities."""
  if not isinstance(agent, LlmAgent):
    return None

  if (
      hasattr(agent, 'generate_content_config')
      and agent.generate_content_config
      and hasattr(agent.generate_content_config, 'response_modalities')
  ):
    return agent.generate_content_config.response_modalities

  return None
