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

from typing import List
from typing import Optional
from typing import Union

from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.function_tool import FunctionTool
from ._livekit_call import _call_for_context
from ._livekit_tools import end_call
from ._livekit_tools import send_dtmf
from ._livekit_tools import transfer_call


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

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ):
    """Initializes the toolset.

    Args:
      tool_filter: Which of the tools above to keep, as a list of names or a
        predicate. Pass `tool_filter=["transfer_call", "send_dtmf"]` for an
        agent that should never decide the conversation is over.
      tool_name_prefix: A prefix to prepend to each tool's name, to keep them
        apart from another toolset's.
    """
    super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)
    # Tool objects are stateless, so build them once rather than per turn.
    self._end_call = FunctionTool(end_call)
    self._transfer_call = FunctionTool(transfer_call)
    self._send_dtmf = FunctionTool(send_dtmf)

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    """Returns the tools the call in progress can honor.

    Args:
      readonly_context: The invocation context, which names the session and so
        identifies the call. None outside an invocation, where there is no call
        to speak of.

    Returns:
      The applicable tools, which is empty when no call is in progress.
    """
    if readonly_context is None:
      return []
    call = _call_for_context(readonly_context)
    if call is None:
      return []
    tools: list[BaseTool] = [self._end_call]
    if call.sip_participant is not None:
      tools += [self._transfer_call, self._send_dtmf]
    return [
        tool for tool in tools if self._is_tool_selected(tool, readonly_context)
    ]
