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

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import override
from typing_extensions import TypeAlias

from ..code_executors.base_code_executor import BaseCodeExecutor
from ..events.event import Event
from ..examples.base_example_provider import BaseExampleProvider
from ..examples.example import Example
from ..flows.llm_flows.auto_flow import AutoFlow
from ..flows.llm_flows.base_llm_flow import BaseLlmFlow
from ..flows.llm_flows.single_flow import SingleFlow
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..planners.base_planner import BasePlanner
from ..tools.base_tool import BaseTool
from ..tools.function_tool import FunctionTool
from ..tools.tool_context import ToolContext
from .base_agent import BaseAgent
from .callback_context import CallbackContext
from .invocation_context import InvocationContext
from .readonly_context import ReadonlyContext

logger = logging.getLogger(__name__)


BeforeModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmRequest], Optional[LlmResponse]
]
AfterModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmResponse],
    Optional[LlmResponse],
]
BeforeToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext],
    Optional[dict],
]
AfterToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext, dict],
    Optional[dict],
]

InstructionProvider: TypeAlias = Callable[[ReadonlyContext], str]

ToolUnion: TypeAlias = Union[Callable, BaseTool]
ExamplesUnion = Union[list[Example], BaseExampleProvider]


def _convert_tool_union_to_tool(
    tool_union: ToolUnion,
) -> BaseTool:
  return (
      tool_union
      if isinstance(tool_union, BaseTool)
      else FunctionTool(tool_union)
  )


class LlmAgent(BaseAgent):
  """
  ===
  LLM-based Agent for building conversational AI applications  
  ===
  
  The `LlmAgent` (often aliased simply as `Agent`) is a core component in ADK, acting as the "thinking" part of your application. It leverages the power of a Large Language Model (LLM) for reasoning, understanding natural language, making decisions, generating responses, and interacting with tools.
  
  Unlike deterministic [Workflow Agents](https://google.github.io/adk-docs/agents/workflow-agents/) that follow predefined execution paths, `LlmAgent` behavior is non-deterministic. It uses the LLM to interpret instructions and context, deciding dynamically how to proceed, which tools to use (if any), or whether to transfer control to another agent.
  
  Building an effective `LlmAgent` involves defining its identity, clearly guiding its behavior through instructions, and equipping it with the necessary tools and capabilities.  
  
  
  ```python
  Arguments : 
    name: str --------------------------------------------------- : The name of the agent (Required)
    description: str = '' --------------------------------------- : A brief description of the agent (Optional, Recommended for Multi-Agent)
    sub_agents: list[BaseAgent] = list -------------------------- : A list of sub-agents (Optional, Default: empty list)
    before_agent_callback: BeforeAgentCallback | None = None ---- : Callback before agent execution (Optional, Default: None)
    after_agent_callback: AfterAgentCallback | None = None ------ : Callback after agent execution (Optional, Default: None)
    model: str | BaseLlm = '' ----------------------------------- : The model to use for the agent (Required)
    instruction: str | InstructionProvider = '' ----------------- : Instructions for the LLM model (Required)
    global_instruction: str | InstructionProvider = '' ---------- : Instructions for all agents (Optional, Default: '')
    tools: list[ToolUnion] = list, ------------------------------ : A list of tools available to this agent (Optional, Default: empty list)
    generate_content_config: GenerateContentConfig | None = None  : Additional content generation configurations (Optional, Default: None)
    disallow_transfer_to_parent: bool = False ------------------- : Disallow transfer to parent agent (Optional, Default: False)
    disallow_transfer_to_peers: bool = False -------------------- : Disallow transfer to peer agents (Optional, Default: False)
    include_contents: Literal['default', 'none'] = 'default' ---- : Include contents in the model request (Optional, Default: 'default')
    input_schema: type[BaseModel] | None = None ----------------- : Input schema for the agent (Optional, Default: None)
    output_schema: type[BaseModel] | None = None ---------------- : Output schema for the agent (Optional, Default: None)
    output_key: str | None = None ------------------------------- : Key in session state to store the output (Optional, Default: None)
    planner: BasePlanner | None = None -------------------------- : Planner for multi-step reasoning (Optional, Default: None)
    code_executor: BaseCodeExecutor | None = None --------------- : Code executor for executing code blocks (Optional, Default: None)
    examples: ExamplesUnion | None = None ----------------------- : Example provider for the agent (Optional, Default: None) 
    before_model_callback: BeforeModelCallback | None = None ---- : Callback before model execution (Optional, Default: None)
    after_model_callback: AfterModelCallback | None = None ------ : Callback after model execution (Optional, Default: None)
    before_tool_callback: BeforeToolCallback | None = None ------ : Callback before tool execution (Optional, Default: None)
    after_tool_callback: AfterToolCallback | None = None -------- : Callback after tool execution (Optional, Default: None)
  ```
  
  <br>
  
  ## Defining the Agent's Identity and Purpose

  First, you need to establish what the agent *is* and what it's *for*.

  * **`name` (Required):** Every agent needs a unique string identifier. This `name` is crucial for internal operations, especially in multi-agent systems where agents need to refer to or delegate tasks to each other. Choose a descriptive name that reflects the agent's function (e.g., `customer_support_router`, `billing_inquiry_agent`). Avoid reserved names like `user`.

  * **`description` (Optional, Recommended for Multi-Agent):** Provide a concise summary of the agent's capabilities. This description is primarily used by *other* LLM agents to determine if they should route a task to this agent. Make it specific enough to differentiate it from peers (e.g., "Handles inquiries about current billing statements," not just "Billing agent").

  * **`model` (Required):** Specify the underlying LLM that will power this agent's reasoning. This is a string identifier like `"gemini-2.0-flash-exp"`. The choice of model impacts the agent's capabilities, cost, and performance. 
  * See the [Models](https://google.github.io/adk-docs/agents/models/) page for available options and considerations.

  ```python
  # Example: Defining the basic identity
  capital_agent = LlmAgent(
      model="gemini-2.0-flash-exp",
      name="capital_agent",
      description="Answers user questions about the capital city of a given country."
      # instruction and tools will be added next
  )
  ```
  <br>
  
  ## Guiding the Agent: Instructions (`instruction`)

  The `instruction` parameter is arguably the most critical for shaping an `LlmAgent`'s behavior. It's a string (or a function returning a string) that tells the agent:

  * Its core task or goal.
  * Its personality or persona (e.g., "You are a helpful assistant," "You are a witty pirate").
  * Constraints on its behavior (e.g., "Only answer questions about X," "Never reveal Y").
  * How and when to use its `tools`. You should explain the purpose of each tool and the circumstances under which it should be called, supplementing any descriptions within the tool itself.
  * The desired format for its output (e.g., "Respond in JSON," "Provide a bulleted list").

  **Tips for Effective Instructions:**

  * **Be Clear and Specific:** Avoid ambiguity. Clearly state the desired actions and outcomes.
  * **Use Markdown:** Improve readability for complex instructions using headings, lists, etc.
  * **Provide Examples (Few-Shot):** For complex tasks or specific output formats, include examples directly in the instruction.
  * **Guide Tool Use:** Don't just list tools; explain *when* and *why* the agent should use them.

  ```python
  # Example: Adding instructions
  capital_agent = LlmAgent(
      model="gemini-2.0-flash-exp",
      name="capital_agent",
      description="Answers user questions about the capital city of a given country.",
      instruction=\"\"\"You are an agent that provides the capital city of a country.
      When a user asks for the capital of a country:
      1. Identify the country name from the user's query.
      2. Use the `get_capital_city` tool to find the capital.
      3. Respond clearly to the user, stating the capital city.
      Example Query: "What's the capital of France?"
      Example Response: "The capital of France is Paris."
      \"\"\",
      # tools will be added next
    )
  ```
  *(Note: For instructions that apply to *all* agents in a system, consider using `global_instruction` on the root agent, detailed further in the [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/) section.)*
  
  <br>
  
  ## Equipping the Agent: Tools (`tools`)
  
  Tools give your `LlmAgent` capabilities beyond the LLM's built-in knowledge or reasoning. They allow the agent to interact with the outside world, perform calculations, fetch real-time data, or execute specific actions.

  * **`tools` (Optional):** Provide a list of tools the agent can use. Each item in the list can be:
      * A Python function (automatically wrapped as a `FunctionTool`).
      * An instance of a class inheriting from `BaseTool`.
      * An instance of another agent (`AgentTool`, enabling agent-to-agent delegation - see [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/)).

  The LLM uses the function/tool names, descriptions (from docstrings or the `description` field), and parameter schemas to decide which tool to call based on the conversation and its instructions.

  ```python
  # Define a tool function
  def get_capital_city(country: str) -> str:
    \"\"\"Retrieves the capital city for a given country.\"\"\"
    # Replace with actual logic (e.g., API call, database lookup)
    capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
    return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

  # Add the tool to the agent
  capital_agent = LlmAgent(
      model="gemini-2.0-flash-exp",
      name="capital_agent",
      description="Answers user questions about the capital city of a given country.",
      instruction=\"\"\"You are an agent that provides the capital city of a country... (previous instruction text)\"\"\",
      tools=[get_capital_city] # Provide the function directly
  )
  ```

  Learn more about Tools in the [Tools](https://google.github.io/adk-docs/tools/) section.
  
  <br>
  
  ## Advanced Configuration & Control

  Beyond the core parameters, `LlmAgent` offers several options for finer control:

  ### Fine-Tuning LLM Generation (`generate_content_config`)

  You can adjust how the underlying LLM generates responses using `generate_content_config`.

  * **`generate_content_config` (Optional):** Pass an instance of `google.genai.types.GenerateContentConfig` to control parameters like `temperature` (randomness), `max_output_tokens` (response length), `top_p`, `top_k`, and safety settings.

      ```python
      from google.genai import types

      agent = LlmAgent(
          # ... other params
          generate_content_config=types.GenerateContentConfig(
              temperature=0.2, # More deterministic output
              max_output_tokens=250
          )
      )
      ```

  ### Structuring Data (`input_schema`, `output_schema`, `output_key`)

  For scenarios requiring structured data exchange, you can use Pydantic models.

  * **`input_schema` (Optional):** Define a Pydantic `BaseModel` class representing the expected input structure. If set, the user message content passed to this agent *must* be a JSON string conforming to this schema. Your instructions should guide the user or preceding agent accordingly.

  * **`output_schema` (Optional):** Define a Pydantic `BaseModel` class representing the desired output structure. If set, the agent's final response *must* be a JSON string conforming to this schema.
      * **Constraint:** Using `output_schema` enables controlled generation within the LLM but **disables the agent's ability to use tools or transfer control to other agents**. Your instructions must guide the LLM to produce JSON matching the schema directly.

  * **`output_key` (Optional):** Provide a string key. If set, the text content of the agent's *final* response will be automatically saved to the session's state dictionary under this key (e.g., `session.state[output_key] = agent_response_text`). This is useful for passing results between agents or steps in a workflow.

  ```python
  from pydantic import BaseModel, Field

  class CapitalOutput(BaseModel):
      capital: str = Field(description="The capital of the country.")

  structured_capital_agent = LlmAgent(
      # ... name, model, description
      instruction=\"\"\"You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}\"\"\",
      output_schema=CapitalOutput, # Enforce JSON output
      output_key="found_capital"  # Store result in state['found_capital']
      # Cannot use tools=[get_capital_city] effectively here
  )
  ```

  ### Managing Context (`include_contents`)

  Control whether the agent receives the prior conversation history.

  * **`include_contents` (Optional, Default: `'default'`):** Determines if the `contents` (history) are sent to the LLM.
      * `'default'`: The agent receives the relevant conversation history.
      * `'none'`: The agent receives no prior `contents`. It operates based solely on its current instruction and any input provided in the *current* turn (useful for stateless tasks or enforcing specific contexts).

      ```python
      stateless_agent = LlmAgent(
          # ... other params
          include_contents='none'
      )
      ```

  ### Planning & Code Execution

  For more complex reasoning involving multiple steps or executing code:

  * **`planner` (Optional):** Assign a `BasePlanner` instance to enable multi-step reasoning and planning before execution. (See [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/) patterns).
  * **`code_executor` (Optional):** Provide a `BaseCodeExecutor` instance to allow the agent to execute code blocks (e.g., Python) found in the LLM's response. ([See Tools/Built-in tools](https://google.github.io/adk-docs/tools/built-in-tools/)).

  ## Putting It Together: Example
      Here's the complete basic `capital_agent`: [Documentation website](https://google.github.io/adk-docs/agents/llm-agents/#putting-it-together-example)

  _(This example demonstrates the core concepts. More complex agents might incorporate schemas, context control, planning, etc.)_

  ## Related Concepts (Deferred Topics)

  While this page covers the core configuration of `LlmAgent`, several related concepts provide more advanced control and are detailed elsewhere:

  * **Callbacks:** Intercepting execution points (before/after model calls, before/after tool calls) using `before_model_callback`, `after_model_callback`, etc. See [Callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/).
  * **Multi-Agent Control:** Advanced strategies for agent interaction, including planning (`planner`), controlling agent transfer (`disallow_transfer_to_parent`, `disallow_transfer_to_peers`), and system-wide instructions (`global_instruction`). See [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/).
  """
  

  model: Union[str, BaseLlm] = ''
  """The model to use for the agent.

  When not set, the agent will inherit the model from its ancestor.
  """

  instruction: Union[str, InstructionProvider] = ''
  """Instructions for the LLM model, guiding the agent's behavior."""

  global_instruction: Union[str, InstructionProvider] = ''
  """Instructions for all the agents in the entire agent tree.

  global_instruction ONLY takes effect in root agent.

  For example: use global_instruction to make all agents have a stable identity
  or personality.
  """

  tools: list[ToolUnion] = Field(default_factory=list)
  """Tools available to this agent."""

  generate_content_config: Optional[types.GenerateContentConfig] = None
  """The additional content generation configurations.

  NOTE: not all fields are usable, e.g. tools must be configured via `tools`,
  thinking_config must be configured via `planner` in LlmAgent.

  For example: use this config to adjust model temperature, configure safety
  settings, etc.
  """

  # LLM-based agent transfer configs - Start
  disallow_transfer_to_parent: bool = False
  """Disallows LLM-controlled transferring to the parent agent."""
  disallow_transfer_to_peers: bool = False
  """Disallows LLM-controlled transferring to the peer agents."""
  # LLM-based agent transfer configs - End

  include_contents: Literal['default', 'none'] = 'default'
  """Whether to include contents in the model request.

  When set to 'none', the model request will not include any contents, such as
  user messages, tool results, etc.
  """

  # Controlled input/output configurations - Start
  input_schema: Optional[type[BaseModel]] = None
  """The input schema when agent is used as a tool."""
  output_schema: Optional[type[BaseModel]] = None
  """The output schema when agent replies.

  NOTE: when this is set, agent can ONLY reply and CANNOT use any tools, such as
  function tools, RAGs, agent transfer, etc.
  """
  output_key: Optional[str] = None
  """The key in session state to store the output of the agent.

  Typically use cases:
  - Extracts agent reply for later use, such as in tools, callbacks, etc.
  - Connects agents to coordinate with each other.
  """
  # Controlled input/output configurations - End

  # Advance features - Start
  planner: Optional[BasePlanner] = None
  """Instructs the agent to make a plan and execute it step by step.

  NOTE: to use model's built-in thinking features, set the `thinking_config`
  field in `google.adk.planners.built_in_planner`.

  """

  code_executor: Optional[BaseCodeExecutor] = None
  """Allow agent to execute code blocks from model responses using the provided
  CodeExecutor.

  Check out available code executions in `google.adk.code_executor` package.

  NOTE: to use model's built-in code executor, don't set this field, add
  `google.adk.tools.built_in_code_execution` to tools instead.
  """
  # Advance features - End

  # TODO: remove below fields after migration. - Start
  # These fields are added back for easier migration.
  examples: Optional[ExamplesUnion] = None
  # TODO: remove above fields after migration. - End

  # Callbacks - Start
  before_model_callback: Optional[BeforeModelCallback] = None
  """Called before calling the LLM.
  Args:
    callback_context: CallbackContext,
    llm_request: LlmRequest, The raw model request. Callback can mutate the
    request.

  Returns:
    The content to return to the user. When present, the model call will be
    skipped and the provided content will be returned to user.
  """
  after_model_callback: Optional[AfterModelCallback] = None
  """Called after calling LLM.

  Args:
    callback_context: CallbackContext,
    llm_response: LlmResponse, the actual model response.

  Returns:
    The content to return to the user. When present, the actual model response
    will be ignored and the provided content will be returned to user.
  """
  before_tool_callback: Optional[BeforeToolCallback] = None
  """Called before the tool is called.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,

  Returns:
    The tool response. When present, the returned tool response will be used and
    the framework will skip calling the actual tool.
  """
  after_tool_callback: Optional[AfterToolCallback] = None
  """Called after the tool is called.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,
    tool_response: The response from the tool.

  Returns:
    When present, the returned dict will be used as tool result.
  """
  # Callbacks - End

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async for event in self._llm_flow.run_async(ctx):
      self.__maybe_save_output_to_state(event)
      yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async for event in self._llm_flow.run_live(ctx):
      self.__maybe_save_output_to_state(event)
      yield event
    if ctx.end_invocation:
      return

  @property
  def canonical_model(self) -> BaseLlm:
    """The resolved self.model field as BaseLlm.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.model, BaseLlm):
      return self.model
    elif self.model:  # model is non-empty str
      return LLMRegistry.new_llm(self.model)
    else:  # find model from ancestors.
      ancestor_agent = self.parent_agent
      while ancestor_agent is not None:
        if isinstance(ancestor_agent, LlmAgent):
          return ancestor_agent.canonical_model
        ancestor_agent = ancestor_agent.parent_agent
      raise ValueError(f'No model found for {self.name}.')

  def canonical_instruction(self, ctx: ReadonlyContext) -> str:
    """The resolved self.instruction field to construct instruction for this agent.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.instruction, str):
      return self.instruction
    else:
      return self.instruction(ctx)

  def canonical_global_instruction(self, ctx: ReadonlyContext) -> str:
    """The resolved self.instruction field to construct global instruction.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.global_instruction, str):
      return self.global_instruction
    else:
      return self.global_instruction(ctx)

  @property
  def canonical_tools(self) -> list[BaseTool]:
    """The resolved self.tools field as a list of BaseTool.

    This method is only for use by Agent Development Kit.
    """
    return [_convert_tool_union_to_tool(tool) for tool in self.tools]

  @property
  def _llm_flow(self) -> BaseLlmFlow:
    if (
        self.disallow_transfer_to_parent
        and self.disallow_transfer_to_peers
        and not self.sub_agents
    ):
      return SingleFlow()
    else:
      return AutoFlow()

  def __maybe_save_output_to_state(self, event: Event):
    """Saves the model output to state if needed."""
    if (
        self.output_key
        and event.is_final_response()
        and event.content
        and event.content.parts
    ):
      result = ''.join(
          [part.text if part.text else '' for part in event.content.parts]
      )
      if self.output_schema:
        result = self.output_schema.model_validate_json(result).model_dump(
            exclude_none=True
        )
      event.actions.state_delta[self.output_key] = result

  @model_validator(mode='after')
  def __model_validator_after(self) -> LlmAgent:
    self.__check_output_schema()
    return self

  def __check_output_schema(self):
    if not self.output_schema:
      return

    if (
        not self.disallow_transfer_to_parent
        or not self.disallow_transfer_to_peers
    ):
      logger.warning(
          'Invalid config for agent %s: output_schema cannot co-exist with'
          ' agent transfer configurations. Setting'
          ' disallow_transfer_to_parent=True, disallow_transfer_to_peers=True',
          self.name,
      )
      self.disallow_transfer_to_parent = True
      self.disallow_transfer_to_peers = True

    if self.sub_agents:
      raise ValueError(
          f'Invalid config for agent {self.name}: if output_schema is set,'
          ' sub_agents must be empty to disable agent transfer.'
      )

    if self.tools:
      raise ValueError(
          f'Invalid config for agent {self.name}: if output_schema is set,'
          ' tools must be empty'
      )

  @field_validator('generate_content_config', mode='after')
  @classmethod
  def __validate_generate_content_config(
      cls, generate_content_config: Optional[types.GenerateContentConfig]
  ) -> types.GenerateContentConfig:
    if not generate_content_config:
      return types.GenerateContentConfig()
    if generate_content_config.thinking_config:
      raise ValueError('Thinking config should be set via LlmAgent.planner.')
    if generate_content_config.tools:
      raise ValueError('All tools must be set via LlmAgent.tools.')
    if generate_content_config.system_instruction:
      raise ValueError(
          'System instruction must be set via LlmAgent.instruction.'
      )
    if generate_content_config.response_schema:
      raise ValueError(
          'Response schema must be set via LlmAgent.output_schema.'
      )
    return generate_content_config


Agent: TypeAlias = LlmAgent
