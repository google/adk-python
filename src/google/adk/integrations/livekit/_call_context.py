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

`Runner.run_live()` takes ids and a queue, so a tool has no parameter through
which to reach the room. `LiveKitRunner` publishes the call on a `ContextVar`
instead, which ADK copies when it dispatches a tool.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import contextvars
import logging
from types import ModuleType
from typing import Optional

from ._rtc import rtc

logger = logging.getLogger("google_adk." + __name__)

# Set by LiveKit on a telephony caller. Every read is optional: the number is
# absent when the dispatch rule hides it, and header-mapped attributes arrive
# asynchronously.
SIP_PHONE_NUMBER_ATTRIBUTE = "sip.phoneNumber"
SIP_TRUNK_PHONE_NUMBER_ATTRIBUTE = "sip.trunkPhoneNumber"
SIP_CALL_ID_ATTRIBUTE = "sip.callID"
SIP_CALL_STATUS_ATTRIBUTE = "sip.callStatus"

_PARTICIPANT_KIND_SIP = rtc.ParticipantKind.PARTICIPANT_KIND_SIP


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
    for participant in self.room.remote_participants.values():
      if participant.kind == _PARTICIPANT_KIND_SIP:
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
    return await self.room.local_participant.perform_rpc(
        destination_identity=identity,
        method=method,
        payload=payload,
        response_timeout=response_timeout,
    )

  def _sole_remote_identity(self) -> str:
    identities = list(self.room.remote_participants)
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
  return api


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

_CURRENT_CALL: contextvars.ContextVar[LiveKitCall] = contextvars.ContextVar(
    "google_adk_livekit_current_call"
)


def _current_call_or_none() -> Optional[LiveKitCall]:
  """Returns the in-progress call, or None when there is no LiveKit session."""
  return _CURRENT_CALL.get(None)


def current_call() -> LiveKitCall:
  """Returns the LiveKit call the calling tool is running inside.

  Returns:
    The in-progress call.

  Raises:
    RuntimeError: If no LiveKit call is in progress.
  """
  call = _CURRENT_CALL.get(None)
  if call is None:
    raise RuntimeError(
        "No LiveKit call is in progress. `current_call()` only works inside an"
        " agent driven by `LiveKitRunner`; this agent is running without a"
        " LiveKit transport."
    )
  return call


@contextlib.contextmanager
def _use_call(call: LiveKitCall) -> Iterator[None]:
  """Publishes `call` to tools for the duration of the block.

  Must wrap the task that drives `run_live`, since ADK snapshots the ambient
  context when it dispatches a tool.
  """
  token = _CURRENT_CALL.set(call)
  try:
    yield
  finally:
    _CURRENT_CALL.reset(token)
