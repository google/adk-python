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

"""Router agent implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, ClassVar, Dict, Type

from pydantic import Field
from typing_extensions import override

from ..events.event import Event
from ..features import experimental
from ..features import FeatureName
from ..utils.context_utils import Aclosing
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .invocation_context import InvocationContext
from .router_agent_config import RouterAgentConfig

logger = logging.getLogger('google_adk.' + __name__)


@experimental(FeatureName.AGENT_STATE)
class RouterAgentState(BaseAgentState):
  """State for RouterAgent."""

  current_route: str = ''
  """The targeted sub-agent name after classification."""

  classifier_finished: bool = False
  """Whether the classifier has completed executing."""


class RouterAgent(BaseAgent):
  """An agent that routes to a specific sub-agent based on a classifier's JSON
  output.

  The RouterAgent operates in two phases:
    1. Classification: Runs a designated classifier sub-agent which outputs
       JSON containing a routing key.
    2. Delegation: Parses the classifier output and delegates execution to
       the matched target sub-agent.

  Example usage:
    ```python
    router = RouterAgent(
        name="my_router",
        classifier_agent_name="classifier",
        routing_key="intent",
        routes={
            "diet": "diet_specialist",
            "habitat": "habitat_specialist",
        },
        default_route="diet_specialist",
        sub_agents=[classifier, diet_specialist, habitat_specialist],
    )
    ```
  """

  config_type: ClassVar[Type[BaseAgentConfig]] = RouterAgentConfig
  """The config type for this agent."""

  classifier_agent_name: str = Field(default='')
  """The name of the sub-agent that acts as the routing classifier."""

  routing_key: str = Field(default='route')
  """The JSON key to extract the chosen route from the classifier's output."""

  routes: Dict[str, str] = Field(default_factory=dict)
  """A dictionary mapping the extracted route string to the target
  sub-agent name."""

  default_route: str = Field(default='')
  """The fallback sub-agent name if the route key doesn't match any mapping."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if not self.sub_agents:
      return

    # Initialize or resume the execution state from the agent state.
    agent_state = (
        self._load_agent_state(ctx, RouterAgentState) or RouterAgentState()
    )

    # Phase 1: Run the classifier agent to determine the route.
    if not agent_state.classifier_finished:
      classifier_agent = self.find_sub_agent(self.classifier_agent_name)
      if not classifier_agent:
        raise ValueError(
            f"Classifier agent '{self.classifier_agent_name}' not found in"
            ' sub_agents.'
        )

      classifier_output = ''
      async with Aclosing(classifier_agent.run_async(ctx)) as agen:
        async for event in agen:
          # Capture the text output but intentionally do NOT yield it
          # so the internal routing logic remains hidden from the user.
          if event.content and getattr(event.content, 'parts', None):
            for part in event.content.parts:
              if getattr(part, 'text', None):
                classifier_output += part.text

      # Parse the classifier output to find the route.
      target_agent_name = self._parse_classifier_output(classifier_output)

      agent_state.current_route = target_agent_name
      agent_state.classifier_finished = True

      if ctx.is_resumable:
        ctx.set_agent_state(self.name, agent_state=agent_state)
        yield self._create_agent_state_event(ctx)

    # Phase 2: Run the selected route agent.
    target_agent = self.find_sub_agent(agent_state.current_route)
    if not target_agent:
      raise ValueError(
          f"Target route agent '{agent_state.current_route}' was not found"
          ' in sub_agents.'
      )

    async with Aclosing(target_agent.run_async(ctx)) as agen:
      async for event in agen:
        yield event

    if ctx.is_resumable:
      ctx.set_agent_state(self.name, end_of_agent=True)
      yield self._create_agent_state_event(ctx)

  def _parse_classifier_output(self, raw_output: str) -> str:
    """Parses the classifier's raw text output into a target agent name.

    Handles common LLM quirks like wrapping JSON in Markdown code fences.

    Args:
      raw_output: The raw text output from the classifier agent.

    Returns:
      The name of the target sub-agent to route to.

    Raises:
      ValueError: If no route could be determined and no default_route
        is configured.
    """
    try:
      # Clean Markdown JSON blocks if present.
      cleaned_output = raw_output.strip()
      if cleaned_output.startswith('```json'):
        cleaned_output = cleaned_output[7:-3]
      elif cleaned_output.startswith('```'):
        cleaned_output = cleaned_output[3:-3]

      parsed_json = json.loads(cleaned_output.strip())
      route_value = parsed_json.get(self.routing_key, '')

      target_agent_name = self.routes.get(route_value, self.default_route)

      if not target_agent_name:
        raise ValueError(
            f"No route mapped for value '{route_value}' and no"
            ' default_route provided.'
        )

      return target_agent_name

    except json.JSONDecodeError as e:
      logger.error(
          'Router classifier did not output valid JSON. Raw output: %s',
          raw_output,
      )
      if self.default_route:
        return self.default_route
      raise ValueError(
          'Classifier returned malformed JSON and no default_route was'
          f' configured. Raw output: {raw_output}'
      ) from e

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if False:  # pylint: disable=using-constant-test
      yield  # Make this a generator function.
    raise NotImplementedError(
        'Live mode routing is not currently supported for RouterAgent.'
    )

  @override
  @classmethod
  @experimental(FeatureName.AGENT_CONFIG)
  def _parse_config(
      cls: type[RouterAgent],
      config: RouterAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    kwargs['classifier_agent_name'] = config.classifier_agent_name
    kwargs['routing_key'] = config.routing_key
    kwargs['routes'] = config.routes
    kwargs['default_route'] = config.default_route
    return kwargs
