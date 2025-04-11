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

"""LiteLlm Agent class for ADK, enabling integration with any LiteLlm-supported models."""

from typing import List, Literal, Optional, Type, Union

from google.genai import types
from pydantic import BaseModel

from ..code_executors.base_code_executor import BaseCodeExecutor
from ..models.lite_llm import LiteLlm
from ..planners.base_planner import BasePlanner
from .base_agent import AfterAgentCallback, BaseAgent, BeforeAgentCallback
from .llm_agent import (AfterModelCallback, AfterToolCallback,
                        BeforeModelCallback, BeforeToolCallback, ExamplesUnion,
                        LlmAgent)


class LiteLlmAgent(LlmAgent):
    """Agent that uses a Lite LLM for inference.

    This agent extends LlmAgent to work with any model supported by LiteLlm,
    allowing integration with models from various providers like OpenAI, Anthropic,
    Ollama, etc. You can provide either a model identifier string or a configured
    LiteLlm instance.

    Args:
        name: A unique name for the agent.
        model: The Lite LLM that the agent will use. Can be a string in "provider/model" format
            or an initialized LiteLlm instance.
        description: A short description of the agent's purpose.
        instruction: Instructions to set the agent's behavior.
        tools: List of tools the agent can use.
        generate_content_config: Configuration for content generation.
        global_instruction: Global instructions for the agent.
        sub_agents: List of sub-agents.
        before_agent_callback: Callback before agent execution.
        after_agent_callback: Callback after agent execution.
        disallow_transfer_to_parent: Whether to disallow transfer to parent.
        disallow_transfer_to_peers: Whether to disallow transfer to peers.
        include_contents: Content inclusion mode.
        input_schema: Schema for input validation.
        output_schema: Schema for output validation.
        output_key: Key for output.
        planner: Planner for the agent.
        code_executor: Code executor for the agent.
        examples: Examples for few-shot learning.
        before_model_callback: Callback before model execution.
        after_model_callback: Callback after model execution.
        before_tool_callback: Callback before tool execution.
        after_tool_callback: Callback after tool execution.
    """

    def __init__(
        self,
        *,
        name: str,
        model: Union[str, LiteLlm],
        description: str = '',
        instruction: str = '',
        tools: Optional[List[Union[callable, BaseAgent]]] = None,
        generate_content_config: Optional[types.GenerateContentConfig] = None,
        global_instruction: str = '',
        sub_agents: Optional[List[BaseAgent]] = None,
        before_agent_callback: Optional[BeforeAgentCallback] = None,
        after_agent_callback: Optional[AfterAgentCallback] = None,
        disallow_transfer_to_parent: bool = False,
        disallow_transfer_to_peers: bool = False,
        include_contents: Literal['default', 'none'] = 'default',
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        planner: Optional[BasePlanner] = None,
        code_executor: Optional[BaseCodeExecutor] = None,
        examples: Optional[ExamplesUnion] = None,
        before_model_callback: Optional[BeforeModelCallback] = None,
        after_model_callback: Optional[AfterModelCallback] = None,
        before_tool_callback: Optional[BeforeToolCallback] = None,
        after_tool_callback: Optional[AfterToolCallback] = None
    ):
        if isinstance(model, str):
            try:
                model_instance = LiteLlm(model=model)
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize LiteLlm with model '{model}': {str(e)}"
                ) from e
        elif isinstance(model, LiteLlm):
            model_instance = model
        else:
            raise TypeError(
                f"Expected model to be a string or LiteLlm instance, got {type(model)}"
            )

        super().__init__(
            name=name,
            description=description,
            sub_agents=sub_agents or [],
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            model=model_instance,
            instruction=instruction,
            global_instruction=global_instruction,
            tools=tools or [],
            generate_content_config=generate_content_config,
            disallow_transfer_to_parent=disallow_transfer_to_parent,
            disallow_transfer_to_peers=disallow_transfer_to_peers,
            include_contents=include_contents,
            input_schema=input_schema,
            output_schema=output_schema,
            output_key=output_key,
            planner=planner,
            code_executor=code_executor,
            examples=examples,
            before_model_callback=before_model_callback,
            after_model_callback=after_model_callback,
            before_tool_callback=before_tool_callback,
            after_tool_callback=after_tool_callback
        )

    def update_model_parameters(self, **kwargs) -> None:
        """Updates LiteLlm model parameters at runtime.
        
        Args:
            **kwargs: Keyword arguments to pass to the LiteLlm instance.
                These will update the additional arguments used during model calls.
        """
        if not isinstance(self.model, LiteLlm):
            raise TypeError("Agent's model is not a LiteLlm instance")
        
        if hasattr(self.model, '_additional_args') and self.model._additional_args is not None:
            self.model._additional_args.update(kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the LiteLlmAgent."""
        model_info = self.model.__class__.__name__ if self.model else "None"
        return f"LiteLlmAgent(name='{self.name}', model={model_info})"