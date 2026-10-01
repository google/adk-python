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

"""Optional HMAC integrity check for function calls stored in the session.

``ToolCallIntegrityPlugin`` stamps each function call with an HMAC when the
event is emitted, checks every stamp before each run, and lets a tool run only
when its call has a valid stamp. Without the secret key, a modified call cannot
be given a valid stamp.

Usage::

    from google.adk.plugins import ToolCallIntegrityPlugin

    plugin = ToolCallIntegrityPlugin(secret_key=key_bytes)
    app = App(name="my_app", root_agent=agent, plugins=[plugin])
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import hmac
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from ..events._internal_metadata import INTERNAL_METADATA_PREFIX
from ..events._internal_metadata import mark_restored
from ..events._internal_metadata import RESTORED_EVENT_KEY
from ..events.event import Event
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from google.genai import types

  from ..agents.invocation_context import InvocationContext
  from ..sessions.session import Session
  from ..tools.base_tool import BaseTool
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

_HMAC_META_KEY = INTERNAL_METADATA_PREFIX + "fc_hmac"
"""``custom_metadata`` key holding ``{function_call_id: stamp}``."""

_STAMP_PREFIX = "v1:"
"""Prefix of every stamp, so the payload format can change later."""


class ToolCallIntegrityError(ValueError):
  """A function call in the session failed its integrity check."""


def _normalize(value: Any) -> Any:
  """Maps whole-number floats to ints so number round trips verify.

  Some session stores keep numbers as doubles, so an ``int`` argument can come
  back as ``1.0``. Everything else, including ``None``, is kept as is.
  """
  if isinstance(value, float) and value.is_integer():
    return int(value)
  if isinstance(value, dict):
    return {k: _normalize(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_normalize(v) for v in value]
  return value


def _canonical_json(value: Any) -> str:
  return json.dumps(
      _normalize(value),
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
  )


def _json_args(function_call: types.FunctionCall) -> dict[str, Any]:
  """The call's args in the JSON form that the session stores write."""
  # Hashing this form lets values such as datetimes and bytes verify after a
  # round trip through storage.
  args = function_call.model_dump(mode="json", include={"args"}).get("args")
  return args or {}


def _canonical_payload(
    session: Session,
    event: Event,
    function_call: types.FunctionCall,
) -> bytes:
  """The bytes a stamp covers: session, invocation, branch, author and call."""
  return _canonical_json({
      "app_name": session.app_name,
      "user_id": session.user_id,
      "session_id": session.id,
      "invocation_id": event.invocation_id,
      "branch": event.branch,
      "author": event.author,
      "name": function_call.name,
      "id": function_call.id,
      "args": _json_args(function_call),
  }).encode()


def _is_stamp_map(value: Any) -> bool:
  return isinstance(value, dict) and all(
      isinstance(k, str) and isinstance(v, str) and v.isascii()
      for k, v in value.items()
  )


def _is_restored(event: Event) -> bool:
  return bool((event.custom_metadata or {}).get(RESTORED_EVENT_KEY))


def _stamps(event: Event) -> dict[str, str]:
  """Returns the event's stamps, raising if the metadata is malformed."""
  stamps = (event.custom_metadata or {}).get(_HMAC_META_KEY)
  if stamps is not None and not _is_stamp_map(stamps):
    raise ToolCallIntegrityError(
        f"Event {event.id!r} has a malformed integrity stamp."
    )
  return stamps or {}


class ToolCallIntegrityPlugin(BasePlugin):
  """Stamps function calls and verifies them before runs and tool calls.

  ``on_event_callback`` adds ``"v1:" + HMAC-SHA256(key, payload)`` for each
  function call to ``event.custom_metadata["__adk_internal_fc_hmac"]``, and
  raises ``ToolCallIntegrityError`` for a response that repeats a call ID
  before it is stored. ``before_run_callback`` recomputes the HMAC for every
  function call with an ID in the session history, answered or not, and raises
  ``ToolCallIntegrityError`` when a stamp is wrong, malformed, repeated or
  left over from a call that no longer matches. ``before_tool_callback`` lets
  a tool run only when every stored copy of its call verifies and the tool's
  name and arguments match one of them; it always refuses a call with no ID.
  Unstamped calls are rejected unless ``allow_unstamped_calls`` is set, in
  which case they are logged.

  Events without function calls are ignored. Events marked as restored are
  not verified, and their function calls are never executed. ADK marks events
  restored through the create-session API or ``adk run --resume``, and
  ``prepare_restored_event`` marks events that your code copies. Anyone who
  can write to the session store can also set the marker, but
  ``before_tool_callback`` still refuses those events' calls.

  ``PluginManager`` re-raises plugin errors as ``RuntimeError``; the
  ``ToolCallIntegrityError`` is its ``__cause__``.

  Register this plugin before other plugins. ``PluginManager`` stops at the
  first ``on_event_callback`` that returns an event, and a later plugin must
  not change function calls or replace ``custom_metadata``.
  """

  def __init__(
      self,
      secret_key: bytes | Sequence[bytes],
      *,
      allow_unstamped_calls: bool = False,
      name: str = "tool_call_integrity",
  ):
    """Initializes the plugin.

    Args:
      secret_key: The HMAC key, or a list of keys during rotation. The first key
        stamps new calls; every key is accepted when verifying.
      allow_unstamped_calls: Log a warning instead of raising for function calls
        that have no stamp, such as calls stored before the plugin was
        installed. Such calls also run, so this reports tampering that removes a
        stamp but does not prevent it. A stamp that is present but wrong is
        always rejected.
      name: The plugin name.

    Raises:
      ValueError: If ``secret_key`` is empty or contains a key that is not
        non-empty bytes.
    """
    super().__init__(name)
    keys = [secret_key] if isinstance(secret_key, bytes) else list(secret_key)
    if not keys or not all(isinstance(k, bytes) and k for k in keys):
      raise ValueError(
          "secret_key must be non-empty bytes or a non-empty list of them."
      )
    self._keys: list[bytes] = keys
    self._allow_unstamped_calls = allow_unstamped_calls

  @staticmethod
  def prepare_restored_event(event: Event) -> Event:
    """Returns a copy with internal metadata removed and marked as restored.

    Use it on events that your code copies into another session, so that the
    history passes the integrity check. The copy only keeps the history: its
    function calls are never executed.

    Args:
      event: The event to copy. It is not modified.

    Returns:
      The marked copy, ready to append to the new session.
    """
    return mark_restored(event.model_copy(deep=True))

  def _sign(self, payload: bytes) -> str:
    digest = hmac.new(self._keys[0], payload, hashlib.sha256).hexdigest()
    return _STAMP_PREFIX + digest

  def _verify(self, payload: bytes, stamp: str) -> bool:
    if not stamp.startswith(_STAMP_PREFIX):
      return False
    digest = stamp[len(_STAMP_PREFIX) :]
    return any(
        hmac.compare_digest(
            hmac.new(key, payload, hashlib.sha256).hexdigest(), digest
        )
        for key in self._keys
    )

  def _unstamped(self, name: str | None, call_id: str | None) -> None:
    if not self._allow_unstamped_calls:
      raise ToolCallIntegrityError(
          f"{name} call {call_id!r} has no integrity stamp."
      )
    logger.warning("%s call %r has no integrity stamp.", name, call_id)

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Event | None:
    session = invocation_context.session
    calls = [fc for fc in event.get_function_calls() if fc.id]
    # Stamps are keyed by call ID, so a response that repeats an ID cannot be
    # stamped, and before_run_callback would reject the session on every later
    # run. This is hypothetical: model provider APIs emit unique IDs, and ADK
    # fills in missing ones. The response is refused before it is stored, so
    # none of its calls run and the session stays usable. Partial events are
    # neither stored nor executed.
    if not event.partial:
      seen: set[str | None] = set()
      for fc in calls:
        if fc.id in seen:
          raise ToolCallIntegrityError(
              f"Event {event.id!r} repeats function call ID {fc.id!r}."
          )
        seen.add(fc.id)
    stamps = {
        fc.id: self._sign(_canonical_payload(session, event, fc))
        for fc in calls
    }
    if stamps:
      event.custom_metadata = {
          **(event.custom_metadata or {}),
          _HMAC_META_KEY: stamps,
      }
    return None

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> types.Content | None:
    session = invocation_context.session
    for event in session.events:
      calls = event.get_function_calls()
      # A stamp on an event without calls has nothing to execute. Calls in
      # restored events are refused by before_tool_callback instead.
      if not calls or _is_restored(event):
        continue

      stamps = _stamps(event)
      verified: set[str] = set()
      for fc in calls:
        # A call without an ID cannot be stamped. before_tool_callback refuses
        # to run it, so there is nothing to check here.
        if not fc.id:
          continue
        if fc.id not in stamps:
          self._unstamped(fc.name, fc.id)
          continue
        if fc.id in verified:
          raise ToolCallIntegrityError(
              f"Event {event.id!r} has duplicate function call ID {fc.id!r}."
          )
        payload = _canonical_payload(session, event, fc)
        if not self._verify(payload, stamps[fc.id]):
          raise ToolCallIntegrityError(
              f"{fc.name} call {fc.id!r} does not match its integrity stamp."
          )
        verified.add(fc.id)

      # A stamp without a matching call in its event means the call was
      # renamed, re-keyed or removed.
      unmatched = stamps.keys() - verified
      if unmatched:
        raise ToolCallIntegrityError(
            f"Event {event.id!r} has integrity stamps with no matching call:"
            f" {sorted(unmatched)}."
        )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    call_id = tool_context.function_call_id
    # ADK gives every new call an ID before running it, so a call without one
    # was replayed from the session and cannot be matched to its stamp.
    if not call_id:
      raise ToolCallIntegrityError(
          f"{tool.name} call has no ID, so its integrity cannot be checked."
      )
    session = tool_context.session
    stored: list[tuple[Event, types.FunctionCall]] = []
    restored = False
    for event in session.events:
      for fc in event.get_function_calls():
        if fc.id != call_id:
          continue
        if _is_restored(event):
          restored = True
        else:
          stored.append((event, fc))
    if not stored:
      if restored:
        raise ToolCallIntegrityError(
            f"{tool.name} call {call_id!r} comes from restored history and"
            " cannot be executed."
        )
      self._unstamped(tool.name, call_id)
      return None

    # Every stored copy of the call must verify, so a repeated ID cannot
    # borrow the stamp of another copy. Restored copies are never trusted; an
    # unstamped copy is trusted only when allow_unstamped_calls logged it.
    trusted: list[types.FunctionCall] = []
    for event, fc in stored:
      stamp = _stamps(event).get(call_id)
      if stamp is None:
        self._unstamped(tool.name, call_id)
      elif not self._verify(_canonical_payload(session, event, fc), stamp):
        raise ToolCallIntegrityError(
            f"{tool.name} call {call_id!r} does not match its integrity stamp."
        )
      trusted.append(fc)

    # What runs must be one of those calls, whichever copy ADK took the
    # arguments from.
    executed = _canonical_json(
        _json_args(stored[0][1].model_copy(update={"args": tool_args}))
    )
    if not any(
        fc.name == tool.name and _canonical_json(_json_args(fc)) == executed
        for fc in trusted
    ):
      raise ToolCallIntegrityError(
          f"{tool.name} call {call_id!r} arguments do not match its stored"
          " call."
      )
    return None
