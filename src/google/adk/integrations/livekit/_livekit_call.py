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

"""Access to the in-progress LiveKit call from inside an ADK tool.

`Runner.run_live()` takes ids and a queue, so there is no parameter through
which to hand a tool the room. What a tool does get is its `ToolContext`, and
that already names the session the call belongs to, so `LiveKitRunner`
registers the call under `(user_id, session_id)` for the duration and
`current_call()` looks it up from the context.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import logging
from types import ModuleType
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ...agents.readonly_context import ReadonlyContext
  from ...tools.tool_context import ToolContext
  from ._rtc import rtc

logger = logging.getLogger("google_adk." + __name__)

# Set by LiveKit on a telephony caller. Every read is optional: the number is
# absent when the dispatch rule hides it, and header-mapped attributes arrive
# asynchronously.
SIP_PHONE_NUMBER_ATTRIBUTE = "sip.phoneNumber"
SIP_TRUNK_PHONE_NUMBER_ATTRIBUTE = "sip.trunkPhoneNumber"
SIP_CALL_ID_ATTRIBUTE = "sip.callID"
SIP_CALL_STATUS_ATTRIBUTE = "sip.callStatus"


class LiveKitCall:
  """The call an ADK tool is currently running inside.

  Obtained with `current_call()`. One instance per `LiveKitRunner`, valid for
  the life of the call.

  Attributes:
    room: The LiveKit room backing the call.
    user_id: The ADK user id for the session.
    session_id: The ADK session id for the session.
  """

  def __init__(
      self,
      *,
      room: rtc.Room,
      user_id: str,
      session_id: str,
      hang_up_callback: Callable[[], None],
  ):
    self.room = room
    self.user_id = user_id
    self.session_id = session_id
    self._hang_up_callback = hang_up_callback

  @property
  def sip_participant(self) -> Optional[rtc.RemoteParticipant]:
    """The telephony caller in the room, or None on a non-SIP call."""
    sip_kind = _sip_participant_kind()
    for participant in self.room.remote_participants.values():
      if participant.kind == sip_kind:
        return participant
    return None

  @property
  def caller_phone_number(self) -> Optional[str]:
    """The number this call came from.

    Returns:
      The caller's number, or None if this is not a SIP call or the dispatch
      rule hides the number.
    """
    return self.sip_attributes().get(SIP_PHONE_NUMBER_ATTRIBUTE)

  def sip_attributes(self) -> dict[str, str]:
    """Returns every `sip.*` attribute LiveKit set on the caller.

    Empty when the call did not arrive over SIP.
    """
    participant = self.sip_participant
    if participant is None:
      return {}
    return {
        key: value
        for key, value in (participant.attributes or {}).items()
        if key.startswith("sip.")
    }

  async def send_dtmf(self, digits: str) -> None:
    """Plays DTMF tones into the call, for navigating a downstream IVR.

    Args:
      digits: The digits to play, e.g. `"123#"`. Characters outside
        `0-9*#A-D` are skipped.
    """
    for digit in digits:
      # The code and the digit must agree, so normalize both.
      key = digit.upper()
      code = _DTMF_CODES.get(key)
      if code is None:
        logger.warning("Skipping non-DTMF character %r.", digit)
        continue
      await self.room.local_participant.publish_dtmf(code=code, digit=key)

  async def perform_rpc(
      self,
      *,
      method: str,
      payload: str,
      destination_identity: Optional[str] = None,
      response_timeout: Optional[float] = None,
  ) -> str:
    """Calls a method the client registered, and returns what it replied.

    Args:
      method: The method name the client registered.
      payload: The request body, as a string.
      destination_identity: Which participant to call. Defaults to the only
        remote participant.
      response_timeout: Seconds to wait for the client's reply.

    Returns:
      The client's reply.

    Raises:
      RuntimeError: If `destination_identity` is omitted and the room does not
        hold exactly one remote participant.
    """
    identity = destination_identity or self._sole_remote_identity()
    reply: str = await self.room.local_participant.perform_rpc(
        destination_identity=identity,
        method=method,
        payload=payload,
        response_timeout=response_timeout,
    )
    return reply

  def _sole_remote_identity(self) -> str:
    identities: list[str] = list(self.room.remote_participants)
    if len(identities) != 1:
      raise RuntimeError(
          "Cannot infer an RPC destination: the room holds"
          f" {len(identities)} remote participants. Pass"
          " destination_identity explicitly."
      )
    return identities[0]

  async def send_data(
      self, payload: bytes, *, topic: str, reliable: bool = True
  ) -> None:
    """Publishes an arbitrary payload on the room data track.

    Args:
      payload: The bytes to publish.
      topic: The data topic clients filter on.
      reliable: Whether to send reliably. False trades delivery for latency,
        which suits high-frequency telemetry.
    """
    await self.room.local_participant.publish_data(
        payload, topic=topic, reliable=reliable
    )

  async def transfer(self, transfer_to: str) -> None:
    """Cold-transfers the SIP caller to another number or SIP URI.

    Uses LiveKit's server API, which reads `LIVEKIT_URL`, `LIVEKIT_API_KEY`
    and `LIVEKIT_API_SECRET` from the environment.

    Args:
      transfer_to: Destination, as `tel:+15105550100` or a `sip:` URI.

    Raises:
      RuntimeError: If the call did not arrive over SIP.
      ImportError: If `livekit-api` is not installed.
    """
    participant = self.sip_participant
    if participant is None:
      raise RuntimeError(
          "Cannot transfer: this call has no SIP participant. Transfers apply"
          " to telephony calls only."
      )
    api = _server_api()
    async with api.LiveKitAPI() as livekit_api:
      await livekit_api.sip.transfer_sip_participant(
          api.TransferSIPParticipantRequest(
              room_name=self.room.name,
              participant_identity=participant.identity,
              transfer_to=transfer_to,
          )
      )

  async def hang_up(self) -> None:
    """Ends the call, closing the model connection and leaving the room.

    On a SIP call the room is deleted as well, because the phone leg is held
    up by the SIP service rather than by a client. The ADK side ends either
    way, so a room that could not be deleted does not also leave the model
    connection open.

    Raises:
      ImportError: If `livekit-api` is not installed and this is a phone call.
      Exception: Whatever the server API raises if the room cannot be deleted,
        after the local session has already ended.
    """
    try:
      if self.sip_participant is not None:
        await self._close_room()
    finally:
      self._hang_up_callback()

  async def _close_room(self) -> None:
    """Deletes the room, disconnecting every participant.

    A room is one call in this model. Use `livekit_api.room.remove_participant`
    instead if a room of yours outlives the agent.
    """
    api = _server_api()
    async with api.LiveKitAPI() as livekit_api:
      await livekit_api.room.delete_room(
          api.DeleteRoomRequest(room=self.room.name)
      )


def _sip_participant_kind() -> int:
  """Returns LiveKit's SIP participant kind, importing the SDK on demand.

  Kept off the module body so `LiveKitToolset` imports without the media SDK;
  reaching a call at all means `LiveKitRunner` already required it.
  """
  from ._rtc import rtc

  kind: int = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
  return kind


def _server_api() -> ModuleType:
  """Returns the `livekit.api` module, which only server-side calls need.

  Imported lazily so an agent that never transfers or hangs up a phone call
  does not pay for the server SDK.

  Raises:
    ImportError: If `livekit-api` is not installed.
  """
  try:
    from livekit import api
  except ImportError as e:
    raise ImportError(
        "livekit-api is not installed. It is required for call transfers and"
        " for ending a phone call. Install it with `pip install"
        ' "google-adk[livekit]"`.'
    ) from e
  module: ModuleType = api
  return module


# RFC 4733 event codes.
_DTMF_CODES: dict[str, int] = {
    **{str(digit): digit for digit in range(10)},
    "*": 10,
    "#": 11,
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
}

# The calls this process is currently serving, keyed by the ADK session each
# one drives. A session is the natural key because it is the only thing both
# sides already know: `LiveKitRunner` is constructed with it, and a tool reads
# it back off its context.
_ACTIVE_CALLS: dict[tuple[str, str], LiveKitCall] = {}


def _call_for_context(context: ReadonlyContext) -> Optional[LiveKitCall]:
  """Returns the call this context belongs to, or None if there is no call."""
  return _ACTIVE_CALLS.get((context.user_id, context.session.id))


def current_call(tool_context: ToolContext) -> LiveKitCall:
  """Returns the LiveKit call the calling tool is running inside.

  Args:
    tool_context: The tool's context, which names the session the call is on.

  Returns:
    The in-progress call.

  Raises:
    RuntimeError: If no LiveKit call is in progress for this session.
  """
  call = _call_for_context(tool_context)
  if call is None:
    raise RuntimeError(
        "No LiveKit call is in progress. `current_call()` only works inside an"
        " agent driven by `LiveKitRunner`; this agent is running without a"
        " LiveKit transport."
    )
  return call


@contextlib.contextmanager
def _register_call(call: LiveKitCall) -> Iterator[None]:
  """Publishes `call` to the tools on its session, for the block's duration.

  Args:
    call: The call to publish.

  Yields:
    None, with the call registered.

  Raises:
    RuntimeError: If a call is already registered on the same session, which
      would otherwise leave the two silently sharing a room.
  """
  key = (call.user_id, call.session_id)
  if key in _ACTIVE_CALLS:
    raise RuntimeError(
        f"A LiveKit call is already in progress for session {call.session_id!r}"
        f" of user {call.user_id!r}. Give each concurrent call its own session"
        " id."
    )
  _ACTIVE_CALLS[key] = call
  try:
    yield
  finally:
    _ACTIVE_CALLS.pop(key, None)
