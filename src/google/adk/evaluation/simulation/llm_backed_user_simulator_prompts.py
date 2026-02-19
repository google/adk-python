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

import textwrap
from typing import Optional

from .user_simulator_personas import UserPersona

_DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE = """You are a Simulated User designed to test an AI Agent.

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
* Output `{{ stop_signal }}` to indicate that the conversation with the AI Agents is over.

# Conversation Plan

{{ conversation_plan }}

# Conversation History

{{ conversation_history }}
"""

_USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE = """
You are a Simulated User designed to test an AI Agent.

Your single most important job is to react logically to the Agent's last message while role-playing as the given Persona.
The Conversation Plan is your canonical grounding, not a script; your response MUST be dictated by what the Agent just said.

# Persona Description

{{ persona.description }}
This persona behaves in the following ways:
{% for b in persona.behaviors %}
## {{ b.name | render_string_filter}}
{{ b.description | render_string_filter }}

Instructions:
{{ b.get_behavior_instructions_str() | render_string_filter }}
{% endfor %}
# Conversation Plan

{{ conversation_plan }}

# Conversation History

{{ conversation_history }}
""".strip()


def is_valid_user_simulator_template(
    template_str: str, required_params: list[str]
) -> bool:
  """Checks if the given template_str is a valid jinja template."""
  from jinja2 import exceptions
  from jinja2 import meta
  from jinja2 import StrictUndefined
  from jinja2.sandbox import SandboxedEnvironment

  # StrictUndefined allows us to check for all the given params.
  env = SandboxedEnvironment(undefined=StrictUndefined)
  try:
    # Check syntax of template
    template = env.parse(template_str)

    # Find all variables the template expects
    undeclared_variables = meta.find_undeclared_variables(template)

    # Check parameters in template
    missing_required = [
        v for v in required_params if v not in undeclared_variables
    ]

    return not (missing_required)

  except (
      exceptions.TemplateSyntaxError,
      exceptions.UndefinedError,
  ) as _:
    return False


def _get_user_simulator_instructions_template(
    custom_instructions: Optional[str] = None,
    user_persona: Optional[UserPersona] = None,
) -> str:
  """Returns the appropriate instruction template for the user simulator."""
  if custom_instructions is None and user_persona is None:
    return _DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE

  if custom_instructions is None and user_persona is not None:
    return _USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE

  if custom_instructions is not None and user_persona is None:
    return custom_instructions

  if custom_instructions is not None and user_persona is not None:
    if not is_valid_user_simulator_template(
        custom_instructions,
        required_params=[
            "stop_signal",
            "conversation_plan",
            "conversation_history",
            "persona",
        ],
    ):
      raise ValueError(
          textwrap.dedent(
              """Custom instructions using personas must contain the following formatting placeholders following Jinja syntax:
                * {{ stop_signal }} : text to be generated when the user simulator decides that the
                  conversation is over.
                * {{ conversation_plan }} : the overall plan for the conversation that the user
                  simulator must follow.
                * {{ conversation_history }} : the conversation between the user and the agent so far.
                * {{ persona }} : UserPersona for the simulator to use.
              """
          )
      )

    return custom_instructions


def get_llm_backed_user_simulator_prompt(
    conversation_plan: str,
    conversation_history: str,
    stop_signal: str,
    custom_instructions: Optional[str] = None,
    user_persona: Optional[UserPersona] = None,
):
  """Formats the prompt for the llm-backed user simulator"""
  from jinja2 import DictLoader
  from jinja2 import pass_context
  from jinja2 import Template
  from jinja2.sandbox import SandboxedEnvironment

  templates = {
      "user_instructions": _get_user_simulator_instructions_template(
          custom_instructions=custom_instructions,
          user_persona=user_persona,
      ),
  }
  template_env = SandboxedEnvironment(loader=DictLoader(templates))

  @pass_context
  def _render_string_filter(context, template_string):
    if not template_string:
      return ""
    return Template(template_string).render(context)

  template_env.filters["render_string_filter"] = _render_string_filter

  template_parameters = {
      "stop_signal": stop_signal,
      "conversation_plan": conversation_plan,
      "conversation_history": conversation_history,
  }
  if user_persona is not None:
    template_parameters["persona"] = user_persona

  return template_env.get_template("user_instructions").render(
      template_parameters
  )
