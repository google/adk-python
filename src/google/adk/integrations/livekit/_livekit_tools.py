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

"""Tools that let a live agent act on the call it is on.

The docstrings below are what the model reads, so they address the model
rather than the developer. `tool_context` is supplied by ADK and dropped from
the declaration the model sees, so it is deliberately absent from the `Args:`
sections.
"""

from __future__ import annotations

import logging

from ...tools.tool_context import ToolContext
from ._livekit_call import current_call

logger = logging.getLogger("google_adk." + __name__)


async def end_call(tool_context: ToolContext) -> str:
  """Hangs up and ends the conversation.

  Call this once the caller has said goodbye or their request is resolved.

  Returns:
    A confirmation that the call is ending.
  """
  call = current_call(tool_context)
  logger.info("Agent ending call for session %s.", call.session_id)
  try:
    await call.hang_up()
  except Exception as e:  # pylint: disable=broad-except
    # `hang_up` has already ended the ADK side, so report the failure rather
    # than failing the turn.
    logger.warning("Hangup for session %s failed: %s", call.session_id, e)
    return f"Could not hang up cleanly: {e}. Ending the conversation anyway."
  return "The call is ending."


async def transfer_call(transfer_to: str, tool_context: ToolContext) -> str:
  """Transfers the caller to another phone number or SIP address.

  Only works on a phone call. Tell the caller they are being transferred
  before calling this, because the transfer takes effect immediately.

  Args:
    transfer_to: Where to send the caller, as a phone number in E.164 form
      (for example `+15105550100`) or a full SIP URI.

  Returns:
    A confirmation, or an explanation of why the transfer could not happen.
  """
  call = current_call(tool_context)
  destination = (
      transfer_to
      if transfer_to.startswith(("tel:", "sip:"))
      else f"tel:{transfer_to}"
  )
  try:
    await call.transfer(destination)
  except (RuntimeError, ImportError) as e:
    # Not a phone call, or no server SDK. Reported back so the model can say so.
    logger.warning("Transfer refused for session %s: %s", call.session_id, e)
    return f"Could not transfer the call: {e}"
  return f"Transferring the caller to {transfer_to}."


async def send_dtmf(digits: str, tool_context: ToolContext) -> str:
  """Presses keys on the phone keypad, to drive an automated phone menu.

  Args:
    digits: The keys to press, in order, for example `"1"` or `"1234#"`.

  Returns:
    A confirmation of what was sent.
  """
  call = current_call(tool_context)
  await call.send_dtmf(digits)
  return f"Sent the tones {digits}."
