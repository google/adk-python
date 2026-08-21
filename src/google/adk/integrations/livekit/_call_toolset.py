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

"""A toolset offering the call tools the current call can honor."""

from __future__ import annotations

from typing import Any
from typing import Optional

from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.function_tool import FunctionTool
from ._call_context import _current_call_or_none
from ._call_tools import end_call
from ._call_tools import send_dtmf
from ._call_tools import transfer_call


@experimental(FeatureName.LIVEKIT)
class LiveKitToolset(BaseToolset):
  """Exposes the call tools that make sense for the call in progress.

  Add it to an agent once and let the transport decide what is offered::

      root_agent = Agent(
          model="gemini-live-2.5-flash-native-audio",
          instruction="...",
          tools=[check_line_status, LiveKitToolset()],
      )

  Resolution happens per invocation:

  | Call in progress          | Tools offered                          |
  | :------------------------ | :------------------------------------- |
  | None, e.g. under adk web  | nothing; the agent runs unchanged      |
  | WebRTC                    | end_call                               |
  | SIP                       | end_call, transfer_call, send_dtmf     |

  Transfers and DTMF go to a SIP peer, so they are meaningless on WebRTC.
  """

  def __init__(self, **kwargs: Any):
    """Initializes the toolset.

    Args:
      **kwargs: Passed to `BaseToolset`. Use `tool_filter` to withhold a tool,
        for example `tool_filter=["transfer_call", "send_dtmf"]` for an agent
        that should never decide the conversation is over.
    """
    super().__init__(**kwargs)
    # Tool objects are stateless, so build them once rather than per turn.
    self._end_call = FunctionTool(end_call)
    self._transfer_call = FunctionTool(transfer_call)
    self._send_dtmf = FunctionTool(send_dtmf)

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    """Returns the tools the call in progress can honor.

    Args:
      readonly_context: The invocation context. Only `tool_filter` reads it;
        the call itself comes from the ambient context `LiveKitRunner`
        publishes, which is set for the whole live session rather than per
        turn.

    Returns:
      The applicable tools, which is empty when no call is in progress.
    """
    call = _current_call_or_none()
    if call is None:
      return []
    tools: list[BaseTool] = [self._end_call]
    if call.sip_participant is not None:
      tools += [self._transfer_call, self._send_dtmf]
    return [
        tool for tool in tools if self._is_tool_selected(tool, readonly_context)
    ]
