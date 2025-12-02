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
from typing import Optional

from typing_extensions import override

from ...events.event import Event
from ...utils.feature_decorator import experimental
from ..eval_case import StaticConversation
from ..evaluator import Evaluator
from .user_simulator import BaseUserSimulatorConfig
from .user_simulator import NextUserMessage
from .user_simulator import Status
from .user_simulator import UserSimulator

logger = logging.getLogger("google_adk." + __name__)


@experimental
class StaticUserSimulator(UserSimulator):
  """A UserSimulator that returns a static list of user messages."""

  def __init__(self, *, static_conversation: StaticConversation):
    super().__init__(
        BaseUserSimulatorConfig(), config_type=BaseUserSimulatorConfig
    )
    self.static_conversation = static_conversation
    self.invocation_idx = 0

  @override
  async def get_next_user_message(
      self,
      events: list[Event],
  ) -> NextUserMessage:
    """Returns the next user message to send to the agent from a static list.

    Args:
      events: The unaltered conversation history between the user and the
        agent(s) under evaluation.

    Returns:
      A NextUserMessage object containing the next user message to send to the
      agent, or a status indicating why no message was generated.
    """
    # check if we have reached the end of the list of invocations
    if self.invocation_idx >= len(self.static_conversation):
      return NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)

    # return the next message in the static list
    next_user_content = self.static_conversation[
        self.invocation_idx
    ].user_content
    self.invocation_idx += 1
    return NextUserMessage(
        status=Status.SUCCESS,
        user_message=next_user_content,
    )

  @override
  def get_simulation_evaluator(
      self,
  ) -> Optional[Evaluator]:
    """The StaticUserSimulator does not require an evaluator."""
    return None
