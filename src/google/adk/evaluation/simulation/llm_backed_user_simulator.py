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
from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from pydantic import Field
from typing_extensions import override

from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...models.registry import LLMRegistry
from ...utils.context_utils import Aclosing
from ...utils.feature_decorator import experimental
from .._retry_options_utils import add_default_retry_options_if_not_present
from ..conversation_scenarios import ConversationScenario
from ..evaluator import Evaluator
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import NextUserMessage
from .user_simulator import Status
from .user_simulator import UserSimulator

logger = logging.getLogger("google_adk." + __name__)

_AUTHOR_USER = "user"
_STOP_SIGNAL = "</finished>"

_USER_AGENT_INSTRUCTIONS_TEMPLATE = """You are a Simulated User designed to test an AI Agent.

Your single most important job is to react logically to the Agent's last message.
The Conversation Plan is your canonical grounding, not a script; your response MUST be dictated by what the Agent just said.

# Primary Operating Loop

You MUST follow this three-step process while thinking:

Step 1: Analyze what the Agent just said or did. Specifically, is the Agent asking you a question, reporting a successful or unsuccessful operation, or saying something incorrect or unexpected?

Step 2: Choose one action based on your analysis:
* ANSWER any questions the Agent asked.
* ADVANCE to the next request as per the Conversation Plan if the Agent succeeds in satisfying your current request.
* INTERVENE if the Agent is yet to complete your current request and the Conversation Plan requires you to modify it.
* CORRECT the Agent if it is making a mistake or failing.
* END the conversation if any of the below stopping conditions are met:
  - The Agent has completed all your requests from the Conversation Plan.
  - The Agent has failed to fulfill a request *more than once*.
  - The Agent has performed an incorrect operation and informs you that it is unable to correct it.
  - The Agent ends the conversation on its own by transferring you to a *human/live agent* (NOT another AI Agent).

Step 3: Formulate a response based on the chosen action and the below Action Protocols and output it.

# Action Protocols

**PROTOCOL: ANSWER**
* Only answer the Agent's questions using information from the Conversation Plan.
* Do NOT provide any additional information the Agent did not explicitly ask for.
* If you do not have the information requested by the Agent, inform the Agent. Do NOT make up information that is not in the Conversation Plan.
* Do NOT advance to the next request in the Conversation Plan.

**PROTOCOL: ADVANCE**
* Make the next request from the Conversation Plan.
* Skip redundant requests already fulfilled by the Agent.

**PROTOCOL: INTERVENE**
* Change your current request as directed by the Conversation Plan with natural phrasing.

**PROTOCOL: CORRECT**
* Challenge illogical or incorrect statements made by the Agent.
* If the Agent did an incorrect operation, ask the Agent to fix it.
* If this is the FIRST time the Agent failed to satisfy your request, ask the Agent to try again.

**PROTOCOL: END**
* End the conversation only when any of the stopping conditions are met; do NOT end prematurely.
* Output `{stop_signal}` to indicate that the conversation with the AI Agents is over.

# Conversation Plan

{conversation_plan}

# Conversation History

{conversation_history}
"""


class LlmBackedUserSimulatorConfig(BaseUserSimulatorConfig):
  """Contains configurations required by an LLM backed user simulator."""

  model: str = Field(
      default="gemini-2.5-flash",
      description="The model to use for user simulation.",
  )

  model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=10240,
          )
      ),
      description="The configuration for the model.",
  )

  max_allowed_invocations: int = Field(
      default=20,
      description="""Maximum number of invocations allowed by the simulated
interaction. This property allows us to stop a run-off conversation, where the
agent and the user simulator get into a never ending loop. The initial fixed
prompt is also counted as an invocation.

(Not recommended) If you don't want a limit, you can set the value to -1.""",
  )


@experimental
class LlmBackedUserSimulator(UserSimulator):
  """A UserSimulator that uses an LLM to generate messages on behalf of the user."""

  config_type: ClassVar[type[LlmBackedUserSimulatorConfig]] = (
      LlmBackedUserSimulatorConfig
  )

  def __init__(
      self,
      *,
      config: BaseUserSimulatorConfig,
      conversation_scenario: ConversationScenario,
  ):
    super().__init__(config, config_type=LlmBackedUserSimulator.config_type)
    self._conversation_scenario = conversation_scenario
    self._invocation_count = 0
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(self._config.model)
    self._llm = llm_class(model=self._config.model)

  @classmethod
  def _summarize_conversation(
      cls,
      events: list[Event],
  ) -> str:
    """Summarize the conversation to add to the prompt.

    Removes tool calls, responses, and thoughts.

    Args:
      events: The conversation history to rewrite.

    Returns:
      The summarized conversation history as a string.
    """
    rewritten_dialogue = []
    for e in events:
      if not e.content or not e.content.parts:
        continue
      author = e.author
      for part in e.content.parts:
        if part.text and not part.thought:
          rewritten_dialogue.append(f"{author}: {part.text}")

    return "\n\n".join(rewritten_dialogue)

  async def _get_llm_response(
      self,
      rewritten_dialogue: str,
  ) -> str:
    """Sends a user message generation request to the LLM and returns the full response."""
    if self._invocation_count == 0:
      # first invocation - send the static starting prompt
      return self._conversation_scenario.starting_prompt

    user_agent_instructions = _USER_AGENT_INSTRUCTIONS_TEMPLATE.format(
        stop_signal=_STOP_SIGNAL,
        conversation_plan=self._conversation_scenario.conversation_plan,
        conversation_history=rewritten_dialogue,
    )

    llm_request = LlmRequest(
        model=self._config.model,
        config=self._config.model_configuration,
        contents=[
            genai_types.Content(
                parts=[
                    genai_types.Part(text=user_agent_instructions),
                ],
                role=_AUTHOR_USER,
            ),
        ],
    )
    add_default_retry_options_if_not_present(llm_request)

    response = ""
    async with Aclosing(self._llm.generate_content_async(llm_request)) as agen:
      async for llm_response in agen:
        generated_content: genai_types.Content = llm_response.content
        if not generated_content.parts:
          continue
        for part in generated_content.parts:
          if part.text and not part.thought:
            response += part.text
    return response

  @override
  async def get_next_user_message(
      self,
      events: list[Event],
  ) -> NextUserMessage:
    """Returns the next user message to send to the agent with help from a LLM.

    Args:
      events: The unaltered conversation history between the user and the
        agent(s) under evaluation.

    Returns:
      A NextUserMessage object containing the next user message to send to the
      agent, or a status indicating why no message was generated.

    Raises:
      RuntimeError: If the user agent fails to generate a message. This is not a
      valid result for the LLM backed user simulator and is different from the
      NO_MESSAGE_GENERATED status.
    """
    # check invocation limit
    invocation_limit = self._config.max_allowed_invocations
    if invocation_limit >= 0 and self._invocation_count >= invocation_limit:
      logger.warning(
          "LlmBackedUserSimulator invocation limit (%d) reached!",
          invocation_limit,
      )
      return NextUserMessage(status=Status.TURN_LIMIT_REACHED)

    # rewrite events for the user simulator
    rewritten_dialogue = self._summarize_conversation(events)

    # query the LLM for the next user message
    response = await self._get_llm_response(rewritten_dialogue)
    self._invocation_count += 1

    # is the conversation over? (Has the user simulator output the stop signal?)
    if _STOP_SIGNAL.lower() in response.lower():
      logger.info(
          "Stopping user message generation as the stop signal was detected."
      )
      return NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)

    # is the response non-empty?
    if response:
      return NextUserMessage(
          status=Status.SUCCESS,
          # return message as user content
          user_message=genai_types.Content(
              parts=[genai_types.Part(text=response)], role=_AUTHOR_USER
          ),
      )

    # if we are here, the user agent failed to generate a message, which is not
    # a valid result for the LLM backed user simulator.
    raise RuntimeError("Failed to generate a user message")

  @override
  def get_simulation_evaluator(
      self,
  ) -> Optional[Evaluator]:
    """Returns an Evaluator that evaluates if the simulation was successful or not."""
    raise NotImplementedError()
