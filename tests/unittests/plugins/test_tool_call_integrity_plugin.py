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

"""Tests for ToolCallIntegrityPlugin."""

from __future__ import annotations

import datetime
import logging
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import mark_restored
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins._tool_call_integrity_plugin import _HMAC_META_KEY
from google.adk.plugins._tool_call_integrity_plugin import _STAMP_PREFIX
from google.adk.plugins._tool_call_integrity_plugin import ToolCallIntegrityError
from google.adk.plugins._tool_call_integrity_plugin import ToolCallIntegrityPlugin
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

_KEY = b"test-secret-key"
_CONFIRM = "adk_request_confirmation"


def _ctx() -> InvocationContext:
  ctx = mock.Mock(spec=InvocationContext)
  ctx.session = Session(id="session", app_name="app", user_id="user")
  return ctx


def _event(
    *calls: types.FunctionCall, author: str = "agent", text: str = ""
) -> Event:
  parts = [types.Part(function_call=fc) for fc in calls]
  if text:
    parts.append(types.Part(text=text))
  return Event(
      invocation_id="inv",
      author=author,
      content=types.Content(role="model", parts=parts),
  )


def _fc(
    fc_id: str | None = "fc-1",
    name: str = _CONFIRM,
    args: dict[str, Any] | None = None,
) -> types.FunctionCall:
  return types.FunctionCall(
      id=fc_id, name=name, args={"amount": 10} if args is None else args
  )


async def _mint(
    plugin: ToolCallIntegrityPlugin, ctx: InvocationContext, event: Event
) -> Event:
  await plugin.on_event_callback(invocation_context=ctx, event=event)
  ctx.session.events.append(event)
  return event


async def _validate(
    plugin: ToolCallIntegrityPlugin, ctx: InvocationContext
) -> None:
  await plugin.before_run_callback(invocation_context=ctx)


def _call(ctx: InvocationContext, index: int = 0) -> types.FunctionCall:
  return ctx.session.events[index].content.parts[0].function_call


class TestConstructor:
  """Key validation."""

  @pytest.mark.parametrize(
      "key", [b"", [], [b"good", b""], "str-key", [b"good", "str-key"]]
  )
  def test_rejects_bad_keys(self, key: Any):
    with pytest.raises(ValueError, match="non-empty bytes"):
      ToolCallIntegrityPlugin(secret_key=key)

  def test_options_are_keyword_only(self):
    with pytest.raises(TypeError):
      ToolCallIntegrityPlugin(_KEY, True)  # pylint: disable=too-many-function-args


class TestStamping:
  """on_event_callback stamps every function call."""

  @pytest.mark.asyncio
  async def test_stamps_every_function_call(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    event = await _mint(
        plugin,
        _ctx(),
        _event(
            _fc("fc-1"),
            _fc("fc-2", name="adk_request_input"),
            _fc("fc-3", name="transfer_money"),
        ),
    )

    stamps = event.custom_metadata[_HMAC_META_KEY]
    assert set(stamps) == {"fc-1", "fc-2", "fc-3"}
    assert len(set(stamps.values())) == 3
    assert all(stamp.startswith(_STAMP_PREFIX) for stamp in stamps.values())

  @pytest.mark.asyncio
  async def test_ignores_events_without_calls(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    text = await _mint(plugin, _ctx(), _event(text="hello"))

    assert not text.custom_metadata

  @pytest.mark.asyncio
  async def test_keeps_existing_custom_metadata(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    event = _event(_fc())
    event.custom_metadata = {"mine": 1}

    await _mint(plugin, _ctx(), event)

    assert event.custom_metadata["mine"] == 1
    assert _HMAC_META_KEY in event.custom_metadata


class TestVerification:
  """before_run_callback accepts untouched sessions and rejects edits."""

  @pytest.mark.asyncio
  async def test_untouched_session_passes(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(text="before"))
    await _mint(plugin, ctx, _event(_fc(args={"a": 1, "b": None})))
    await _mint(plugin, ctx, _event(text="after"))

    await _validate(plugin, ctx)

  @pytest.mark.parametrize(
      "tamper",
      [
          pytest.param(lambda fc: fc.args.update(amount=10000), id="changed"),
          pytest.param(lambda fc: fc.args.update(extra="x"), id="added"),
          pytest.param(lambda fc: fc.args.update(extra=None), id="added_none"),
          pytest.param(lambda fc: fc.args.pop("amount"), id="removed"),
          pytest.param(lambda fc: setattr(fc, "name", "tool"), id="renamed"),
      ],
  )
  @pytest.mark.parametrize("name", [_CONFIRM, "transfer_money"])
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_modified_call_is_rejected(
      self, tamper, name, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc(name=name)))

    tamper(_call(ctx))

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(plugin, ctx)

  @pytest.mark.parametrize("field", ["invocation_id", "branch", "author"])
  @pytest.mark.asyncio
  async def test_call_moved_to_other_event_context_is_rejected(self, field):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc(name="transfer_money")))

    setattr(ctx.session.events[0], field, "other")

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(plugin, ctx)

  @pytest.mark.parametrize("field", ["id", "app_name", "user_id"])
  @pytest.mark.asyncio
  async def test_stamp_from_other_session_is_rejected(self, field):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    setattr(ctx.session, field, "other")

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_stamp_without_version_prefix_is_rejected(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    event = await _mint(plugin, ctx, _event(_fc()))

    stamps = event.custom_metadata[_HMAC_META_KEY]
    stamps["fc-1"] = stamps["fc-1"].removeprefix(_STAMP_PREFIX)

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(plugin, ctx)

  @pytest.mark.parametrize(
      "tamper",
      [
          pytest.param(lambda fc: setattr(fc, "id", "fc-9"), id="re_keyed"),
          pytest.param(lambda fc: setattr(fc, "id", None), id="id_removed"),
      ],
  )
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_stamp_without_its_call_is_rejected(
      self, tamper, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    tamper(_call(ctx))

    # By default the moved call may first be rejected as unstamped.
    with pytest.raises(
        ToolCallIntegrityError, match="no matching call|no integrity stamp"
    ):
      await _validate(plugin, ctx)

  @pytest.mark.parametrize(
      "stamps",
      ["abc", ["fc-1"], {"fc-1": 1}, {1: "abc"}, {"fc-1": "v1:\u00e9"}],
      ids=repr,
  )
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_malformed_stamp_is_rejected(
      self, stamps, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    event = await _mint(plugin, ctx, _event(_fc()))

    event.custom_metadata[_HMAC_META_KEY] = stamps

    with pytest.raises(ToolCallIntegrityError, match="malformed"):
      await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_whole_number_float_round_trip_passes(self):
    """Stores that keep numbers as doubles turn 10 into 10.0."""
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc(args={"n": 10, "l": [1, {"m": 2}]})))

    _call(ctx).args.update(n=10.0, l=[1.0, {"m": 2.0}])

    await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_reordered_events_pass(self):
    """Stamps are not chained, so event order does not matter."""
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc("fc-1")))
    await _mint(plugin, ctx, _event(_fc("fc-2")))

    ctx.session.events.reverse()

    await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_error_names_the_modified_call(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc("fc-1")))
    await _mint(plugin, ctx, _event(_fc("fc-2")))

    _call(ctx, 1).args["amount"] = 1

    with pytest.raises(ToolCallIntegrityError, match="fc-2"):
      await _validate(plugin, ctx)


class TestUnstampedCalls:
  """allow_unstamped_calls only decides what happens to unstamped calls."""

  @pytest.mark.parametrize(
      "event",
      [
          pytest.param(_event(_fc()), id="never_stamped"),
          pytest.param(
              _event(_fc(name="transfer_money")), id="regular_never_stamped"
          ),
      ],
  )
  @pytest.mark.asyncio
  async def test_rejected_by_default(self, event):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    ctx.session.events.append(event)

    with pytest.raises(ToolCallIntegrityError, match="no integrity stamp"):
      await _validate(plugin, ctx)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_calls_without_id_are_not_checked(self, allow_unstamped_calls):
    """ID-less calls cannot be stamped; the gate refuses to run them instead."""
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    ctx.session.events.append(
        _event(_fc(fc_id=None), _fc(fc_id=None, name="transfer_money"))
    )

    await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_stripped_stamp_is_rejected_by_default(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    event = await _mint(plugin, ctx, _event(_fc()))

    del event.custom_metadata[_HMAC_META_KEY]

    with pytest.raises(ToolCallIntegrityError, match="no integrity stamp"):
      await _validate(plugin, ctx)

  @pytest.mark.asyncio
  async def test_allow_unstamped_calls_warns(self, caplog):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=True
    )
    ctx = _ctx()
    ctx.session.events.append(_event(_fc()))

    with caplog.at_level(logging.WARNING):
      await _validate(plugin, ctx)

    assert "no integrity stamp" in caplog.text


class TestKeyRotation:
  """The first key signs; every key verifies."""

  @pytest.mark.asyncio
  async def test_old_stamps_verify_during_rotation(self):
    ctx = _ctx()
    await _mint(ToolCallIntegrityPlugin(secret_key=b"old"), ctx, _event(_fc()))

    await _validate(ToolCallIntegrityPlugin(secret_key=[b"new", b"old"]), ctx)

  @pytest.mark.asyncio
  async def test_new_key_signs(self):
    ctx = _ctx()
    rotating = ToolCallIntegrityPlugin(secret_key=[b"new", b"old"])
    await _mint(rotating, ctx, _event(_fc()))

    await _validate(ToolCallIntegrityPlugin(secret_key=b"new"), ctx)

  @pytest.mark.asyncio
  async def test_retired_key_is_rejected(self):
    ctx = _ctx()
    await _mint(ToolCallIntegrityPlugin(secret_key=b"old"), ctx, _event(_fc()))

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(ToolCallIntegrityPlugin(secret_key=b"new"), ctx)

  @pytest.mark.asyncio
  async def test_rotation_still_rejects_modified_calls(self):
    ctx = _ctx()
    await _mint(ToolCallIntegrityPlugin(secret_key=b"old"), ctx, _event(_fc()))

    _call(ctx).args["amount"] = 10000

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _validate(ToolCallIntegrityPlugin(secret_key=[b"new", b"old"]), ctx)


class TestDuplicateIds:
  """A call ID may appear once per event."""

  @pytest.mark.parametrize(
      "other_args", [{"amount": 10}, {"amount": 999}], ids=["same", "different"]
  )
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_response_repeating_a_call_id_is_refused_when_emitted(
      self, allow_unstamped_calls, other_args
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    event = _event(_fc("fc-1"), _fc("fc-2"), _fc("fc-1", args=other_args))

    with pytest.raises(ToolCallIntegrityError, match="repeats function call"):
      await plugin.on_event_callback(invocation_context=_ctx(), event=event)

    assert not event.custom_metadata

  @pytest.mark.asyncio
  async def test_partial_response_repeating_a_call_id_is_not_refused(self):
    """Partial events are neither stored nor executed."""
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    event = _event(_fc("fc-1"), _fc("fc-1"))
    event.partial = True

    await plugin.on_event_callback(invocation_context=_ctx(), event=event)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_duplicate_call_id_in_one_event_is_rejected(
      self, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    event = await _mint(plugin, ctx, _event(_fc()))

    event.content.parts.append(event.content.parts[0].model_copy(deep=True))

    with pytest.raises(ToolCallIntegrityError, match="duplicate"):
      await _validate(plugin, ctx)


class TestStoreRoundTrip:
  """Stamps verify after the JSON round trip that session stores perform."""

  def test_plugin_key_uses_reserved_prefix(self):
    assert _HMAC_META_KEY.startswith(INTERNAL_METADATA_PREFIX)

  @pytest.mark.asyncio
  async def test_datetime_request_input_payload_can_be_stamped(self):
    args = RequestInput(
        payload={"when": datetime.datetime(2026, 1, 2, 3, 4, 5)}
    ).model_dump(by_alias=True)
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)

    event = await _mint(
        plugin,
        _ctx(),
        _event(_fc("ri-1", name="adk_request_input", args=args)),
    )

    assert "ri-1" in event.custom_metadata[_HMAC_META_KEY]

  @pytest.mark.parametrize(
      "value",
      [
          datetime.datetime(2026, 1, 2, 3, 4, 5, 123456),
          datetime.date(2026, 1, 2),
          b"\x00\xffbinary",
          {3, 1, 2},
          float("nan"),
          {"nested": [datetime.datetime(2026, 1, 2), b"ab"]},
      ],
      ids=repr,
  )
  @pytest.mark.asyncio
  async def test_stamp_verifies_after_store_round_trip(self, value):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    event = _event(_fc(name="tool", args={"v": value}))
    await plugin.on_event_callback(invocation_context=ctx, event=event)

    ctx.session.events.append(
        Event.model_validate_json(event.model_dump_json(exclude_none=True))
    )

    await _validate(plugin, ctx)

  @pytest.mark.parametrize(
      "stamps", [{"fc-9": _STAMP_PREFIX + "0" * 64}, "malformed"], ids=repr
  )
  @pytest.mark.parametrize("author", ["user", "agent"])
  @pytest.mark.asyncio
  async def test_stamp_on_event_without_calls_is_ignored(self, stamps, author):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    event = _event(author=author, text="hi")
    event.custom_metadata = {_HMAC_META_KEY: stamps}
    ctx.session.events.append(event)

    await _validate(plugin, ctx)


def _set_restored_marker(event: Event) -> Event:
  """Sets the marker the way a store writer could, keeping other metadata."""
  event.custom_metadata = {
      **(event.custom_metadata or {}),
      RESTORED_EVENT_KEY: True,
  }
  return event


class TestRestoredHistory:
  """before_run_callback accepts events marked as restored without checks."""

  @pytest.mark.parametrize(
      "event",
      [
          pytest.param(_event(_fc()), id="unstamped"),
          pytest.param(_event(_fc(fc_id=None)), id="no_id"),
          pytest.param(
              _event(_fc(fc_id=None), _fc(fc_id=None, name="transfer_money")),
              id="several_without_ids",
          ),
          pytest.param(_event(_fc(), _fc()), id="duplicate_ids"),
      ],
  )
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_restored_history_is_accepted(
      self, event, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    ctx.session.events.append(mark_restored(event.model_copy(deep=True)))

    await _validate(plugin, ctx)

  @pytest.mark.parametrize("stamps", [{"fc-1": "v1:bad"}, "malformed"])
  @pytest.mark.asyncio
  async def test_marked_event_is_not_checked(self, stamps):
    """A store writer can set the marker; the gate still refuses its calls."""
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    event = _event(_fc())
    event.custom_metadata = {_HMAC_META_KEY: stamps}
    ctx.session.events.append(_set_restored_marker(event))

    await _validate(plugin, ctx)


def _tool_context(session: Session, call_id: str | None) -> mock.Mock:
  tool_context = mock.Mock()
  tool_context.function_call_id = call_id
  tool_context.session = session
  return tool_context


async def _gate(
    plugin: ToolCallIntegrityPlugin,
    ctx: InvocationContext,
    call_id: str | None = "fc-1",
    tool_args: dict[str, Any] | None = None,
    name: str = _CONFIRM,
) -> dict[str, Any] | None:
  tool = mock.Mock()
  tool.name = name
  return await plugin.before_tool_callback(
      tool=tool,
      tool_args={"amount": 10} if tool_args is None else tool_args,
      tool_context=_tool_context(ctx.session, call_id),
  )


class TestExecutionGate:
  """before_tool_callback lets a tool run only if its stored call verifies."""

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_verified_call_runs(self, allow_unstamped_calls):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    assert await _gate(plugin, ctx) is None

  @pytest.mark.parametrize("stored", [False, True])
  @pytest.mark.asyncio
  async def test_unstamped_call_is_rejected_by_default(self, stored):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    if stored:
      ctx.session.events.append(_event(_fc()))

    with pytest.raises(ToolCallIntegrityError, match="no integrity stamp"):
      await _gate(plugin, ctx)

  @pytest.mark.parametrize("stored", [False, True])
  @pytest.mark.asyncio
  async def test_unstamped_call_warns_and_runs_when_allowed(
      self, stored, caplog
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=True
    )
    ctx = _ctx()
    if stored:
      ctx.session.events.append(_event(_fc()))

    with caplog.at_level(logging.WARNING):
      assert await _gate(plugin, ctx) is None

    assert "no integrity stamp" in caplog.text

  @pytest.mark.parametrize("restored", [False, True])
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_call_without_id_is_rejected(
      self, restored, allow_unstamped_calls
  ):
    """ADK gives fresh calls an ID, so only a replayed stored call lacks one."""
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    event = _event(_fc(fc_id=None))
    ctx.session.events.append(mark_restored(event) if restored else event)

    with pytest.raises(ToolCallIntegrityError, match="no ID"):
      await _gate(plugin, ctx, call_id=None)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_wrong_stamp_is_rejected(self, allow_unstamped_calls):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    _call(ctx).args["amount"] = 10000

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _gate(plugin, ctx)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_restored_call_is_rejected(self, allow_unstamped_calls):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    ctx.session.events.append(mark_restored(_event(_fc())))

    with pytest.raises(ToolCallIntegrityError, match="restored"):
      await _gate(plugin, ctx)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_stamped_call_marked_restored_is_rejected(
      self, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    event = await _mint(plugin, ctx, _event(_fc()))

    _set_restored_marker(event)

    with pytest.raises(ToolCallIntegrityError, match="restored"):
      await _gate(plugin, ctx)

  # A repeated ID cannot borrow the stamp of another copy: every stored copy
  # of the call must verify.

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_modified_copy_of_a_verified_call_is_rejected(
      self, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))
    await _mint(plugin, ctx, _event(_fc()))

    _call(ctx, 1).args["amount"] = 10000

    with pytest.raises(ToolCallIntegrityError, match="does not match"):
      await _gate(plugin, ctx)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_restored_copy_does_not_block_a_verified_call(
      self, allow_unstamped_calls
  ):
    """Providers can reuse IDs, so a restored call can share a new call's ID."""
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))
    ctx.session.events.append(mark_restored(_event(_fc(args={"amount": 5}))))

    assert await _gate(plugin, ctx) is None
    with pytest.raises(ToolCallIntegrityError, match="arguments"):
      await _gate(plugin, ctx, tool_args={"amount": 5})

  # The executed call must match a verified stored call, whichever copy ADK
  # took its arguments from.

  @pytest.mark.parametrize(
      "tool_args, name",
      [({"amount": 10000}, _CONFIRM), ({}, _CONFIRM), ({"amount": 10}, "t")],
      ids=["changed_args", "missing_args", "other_tool"],
  )
  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_executed_call_must_match_the_verified_call(
      self, tool_args, name, allow_unstamped_calls
  ):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    with pytest.raises(ToolCallIntegrityError, match="arguments"):
      await _gate(plugin, ctx, tool_args=tool_args, name=name)

  @pytest.mark.asyncio
  async def test_whole_number_float_args_match_the_verified_call(self):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))

    assert await _gate(plugin, ctx, tool_args={"amount": 10.0}) is None

  @pytest.mark.asyncio
  async def test_unstamped_call_with_other_args_is_refused_when_allowed(self):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=True
    )
    ctx = _ctx()
    ctx.session.events.append(_event(_fc()))

    with pytest.raises(ToolCallIntegrityError, match="arguments"):
      await _gate(plugin, ctx, tool_args={"amount": 10000})

  @pytest.mark.asyncio
  async def test_unstamped_copy_of_a_verified_call_is_rejected_by_default(
      self,
  ):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    ctx = _ctx()
    await _mint(plugin, ctx, _event(_fc()))
    ctx.session.events.append(_event(_fc(args={"amount": 10000})))

    with pytest.raises(ToolCallIntegrityError, match="no integrity stamp"):
      await _gate(plugin, ctx)


class TestPrepareRestoredEvent:
  """Copies prepared for another session keep history but never execute."""

  def test_returns_marked_copy_without_internal_metadata(self):
    event = _event(_fc())
    event.custom_metadata = {_HMAC_META_KEY: {"fc-1": "v1:x"}, "mine": 1}

    copy = ToolCallIntegrityPlugin.prepare_restored_event(event)

    assert copy is not event
    assert copy.custom_metadata == {"mine": 1, RESTORED_EVENT_KEY: True}
    assert event.custom_metadata == {
        _HMAC_META_KEY: {"fc-1": "v1:x"},
        "mine": 1,
    }
    assert copy.content == event.content
    assert copy.content is not event.content

  @pytest.mark.parametrize(
      "source", ["stamped_elsewhere", "unstamped", "no_id"]
  )
  @pytest.mark.asyncio
  async def test_copy_passes_history_check_by_default(self, source):
    plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
    other = _ctx()
    other.session.id = "other-session"
    event = _event(_fc(fc_id=None if source == "no_id" else "fc-1"))
    if source == "stamped_elsewhere":
      await plugin.on_event_callback(invocation_context=other, event=event)
    ctx = _ctx()

    ctx.session.events.append(
        ToolCallIntegrityPlugin.prepare_restored_event(event)
    )

    await _validate(plugin, ctx)

  @pytest.mark.parametrize("allow_unstamped_calls", [False, True])
  @pytest.mark.asyncio
  async def test_copied_call_never_executes(self, allow_unstamped_calls):
    plugin = ToolCallIntegrityPlugin(
        secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
    )
    ctx = _ctx()
    event = _event(_fc())
    await plugin.on_event_callback(invocation_context=ctx, event=event)

    ctx.session.events.append(
        ToolCallIntegrityPlugin.prepare_restored_event(event)
    )

    with pytest.raises(ToolCallIntegrityError, match="restored"):
      await _gate(plugin, ctx)


# End to end, through Runner.run_async and a real confirmation flow.

_APP = "app"
_USER = "user"


class _ScriptedModel(BaseLlm):
  """Returns canned responses in order."""

  model: str = "scripted"
  responses: list[LlmResponse]
  calls: int = 0

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    response = self.responses[self.calls]
    self.calls += 1
    yield response


class _Bank:
  """Owns a confirmation-gated tool and records what it was asked to do."""

  def __init__(self):
    self.transfers: list[dict[str, Any]] = []

  def transfer_money(
      self, amount: int, recipient: str, memo: str | None = None
  ) -> dict[str, Any]:
    """Transfers money."""
    self.transfers.append(
        {"amount": amount, "recipient": recipient, "memo": memo}
    )
    return {"status": "done"}


def _runner(bank: _Bank, session_service: BaseSessionService) -> Runner:
  model = _ScriptedModel(
      responses=[
          LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              name="transfer_money",
                              args={
                                  "amount": 10,
                                  "recipient": "alice",
                                  "memo": None,
                              },
                          )
                      )
                  ],
              )
          ),
          LlmResponse(
              content=types.Content(
                  role="model", parts=[types.Part(text="Sent.")]
              )
          ),
      ]
  )
  agent = LlmAgent(
      name="bank_agent",
      model=model,
      tools=[FunctionTool(bank.transfer_money, require_confirmation=True)],
  )
  app = App(
      name=_APP,
      root_agent=agent,
      plugins=[ToolCallIntegrityPlugin(secret_key=_KEY)],
  )
  return Runner(app=app, session_service=session_service)


async def _run(
    runner: Runner, session_id: str, message: types.Content
) -> list[Event]:
  return [
      event
      async for event in runner.run_async(
          user_id=_USER, session_id=session_id, new_message=message
      )
  ]


async def _request_confirmation(
    runner: Runner, session_service: BaseSessionService
) -> tuple[str, str]:
  """Runs until the agent asks for approval; returns (session, call) IDs."""
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  events = await _run(
      runner,
      session.id,
      types.Content(role="user", parts=[types.Part(text="Pay alice 10")]),
  )
  confirmations = []
  for event in events:
    for fc in event.get_function_calls():
      if fc.name == _CONFIRM:
        confirmations.append(fc)
  assert len(confirmations) == 1
  return session.id, confirmations[0].id


def _approval(call_id: str) -> types.Content:
  return types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=call_id, name=_CONFIRM, response={"confirmed": True}
              )
          )
      ],
  )


def _stored_events(
    session_service: InMemorySessionService, session_id: str
) -> list[Event]:
  return session_service.sessions[_APP][_USER][session_id].events


def _assert_rejected_by_plugin(
    error: RuntimeError, callback: str, match: str
) -> None:
  """PluginManager wraps plugin errors; check the wrapped one is ours."""
  assert isinstance(error.__cause__, ToolCallIntegrityError)
  assert f"'tool_call_integrity' during '{callback}'" in str(error)
  assert match in str(error.__cause__)


@pytest.mark.parametrize("store", ["memory", "sqlite"])
@pytest.mark.asyncio
async def test_e2e_approved_call_runs(store, tmp_path):
  """Stamps survive persistence, and an untouched approval goes through."""
  if store == "memory":
    session_service = InMemorySessionService()
  else:
    from google.adk.sessions.sqlite_session_service import SqliteSessionService  # pylint: disable=g-import-not-at-top

    session_service = SqliteSessionService(str(tmp_path / "sessions.db"))
  bank = _Bank()
  runner = _runner(bank, session_service)
  session_id, call_id = await _request_confirmation(runner, session_service)

  await _run(runner, session_id, _approval(call_id))

  assert bank.transfers == [{"amount": 10, "recipient": "alice", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_modified_call_is_blocked():
  """An attacker who rewrites the pending transfer cannot get it executed."""
  session_service = InMemorySessionService()
  bank = _Bank()
  runner = _runner(bank, session_service)
  session_id, call_id = await _request_confirmation(runner, session_service)

  # Rewrite the amount everywhere it is stored, so ADK's own check that the
  # confirmation matches the original call still passes.
  for event in _stored_events(session_service, session_id):
    for fc in event.get_function_calls():
      if fc.name == "transfer_money":
        fc.args["amount"] = 10000
      elif fc.name == _CONFIRM:
        fc.args["originalFunctionCall"]["args"]["amount"] = 10000

  with pytest.raises(RuntimeError) as raised:
    await _run(runner, session_id, _approval(call_id))

  _assert_rejected_by_plugin(
      raised.value, "before_run_callback", "does not match"
  )
  assert not bank.transfers


@pytest.mark.asyncio
async def test_e2e_modified_original_call_is_blocked():
  """The tool call itself is stamped, not only the confirmation request."""
  session_service = InMemorySessionService()
  bank = _Bank()
  runner = _runner(bank, session_service)
  session_id, call_id = await _request_confirmation(runner, session_service)

  for event in _stored_events(session_service, session_id):
    for fc in event.get_function_calls():
      if fc.name == "transfer_money":
        fc.args["recipient"] = "mallory"

  with pytest.raises(RuntimeError) as raised:
    await _run(runner, session_id, _approval(call_id))

  _assert_rejected_by_plugin(
      raised.value, "before_run_callback", "does not match"
  )
  assert not bank.transfers


@pytest.mark.asyncio
async def test_e2e_stripped_stamp_is_blocked():
  session_service = InMemorySessionService()
  bank = _Bank()
  runner = _runner(bank, session_service)
  session_id, call_id = await _request_confirmation(runner, session_service)

  for event in _stored_events(session_service, session_id):
    if event.custom_metadata:
      event.custom_metadata.pop(_HMAC_META_KEY, None)

  with pytest.raises(RuntimeError) as raised:
    await _run(runner, session_id, _approval(call_id))

  _assert_rejected_by_plugin(
      raised.value, "before_run_callback", "no integrity stamp"
  )
  assert not bank.transfers


@pytest.mark.asyncio
async def test_e2e_approved_call_marked_restored_is_blocked_at_execution():
  """A store writer can set the marker, but the gate still refuses the call."""
  session_service = InMemorySessionService()
  bank = _Bank()
  runner = _runner(bank, session_service)
  session_id, call_id = await _request_confirmation(runner, session_service)

  for event in _stored_events(session_service, session_id):
    if any(fc.name == "transfer_money" for fc in event.get_function_calls()):
      _set_restored_marker(event)

  with pytest.raises(RuntimeError) as raised:
    await _run(runner, session_id, _approval(call_id))

  _assert_rejected_by_plugin(raised.value, "before_tool_callback", "restored")
  assert not bank.transfers


# End to end, in a resumable app where an unanswered trailing call runs again
# on resume.

_AGENT = "resumable_agent"


def _transfer_response(amount: int = 2, recipient: str = "bob") -> LlmResponse:
  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="transfer_money",
                      args={"amount": amount, "recipient": recipient},
                  )
              )
          ],
      )
  )


def _text_response(text: str = "ok") -> LlmResponse:
  return LlmResponse(
      content=types.Content(role="model", parts=[types.Part(text=text)])
  )


def _user_message(text: str = "hi") -> types.Content:
  return types.Content(role="user", parts=[types.Part(text=text)])


def _resumable_runner(
    bank: _Bank,
    plugin: ToolCallIntegrityPlugin | None,
    responses: list[LlmResponse] | None = None,
) -> tuple[Runner, InMemorySessionService]:
  session_service = InMemorySessionService()
  agent = LlmAgent(
      name=_AGENT,
      model=_ScriptedModel(responses=responses or [_text_response()]),
      tools=[FunctionTool(bank.transfer_money)],
  )
  app = App(
      name=_APP,
      root_agent=agent,
      plugins=[plugin] if plugin else [],
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  return Runner(app=app, session_service=session_service), session_service


def _transfer_event(
    fc_id: str | None = "fc-1", invocation_id: str = "inv-1"
) -> Event:
  return Event(
      invocation_id=invocation_id,
      author=_AGENT,
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=_fc(
                      fc_id,
                      name="transfer_money",
                      args={"amount": 1, "recipient": "alice"},
                  )
              )
          ],
      ),
  )


def _transfer_result_event(fc_id: str | None) -> Event:
  return Event(
      invocation_id="old-inv",
      author=_AGENT,
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=fc_id, name="transfer_money", response={"ok": True}
                  )
              )
          ],
      ),
  )


async def _stamped(
    plugin: ToolCallIntegrityPlugin, session: Session, event: Event
) -> Event:
  ctx = mock.Mock(spec=InvocationContext)
  ctx.session = session
  await plugin.on_event_callback(invocation_context=ctx, event=event)
  return event


async def _resume(runner: Runner, session_id: str) -> list[Event]:
  return [
      event
      async for event in runner.run_async(
          user_id=_USER, session_id=session_id, invocation_id="inv-1"
      )
  ]


@pytest.mark.parametrize("fc_id", ["r-1", None], ids=["with_id", "no_id"])
@pytest.mark.parametrize("allow_unstamped_calls", [False, True])
@pytest.mark.asyncio
async def test_e2e_restored_history_with_tool_calls_keeps_working(
    fc_id, allow_unstamped_calls
):
  """Restored history loads, including calls without IDs; nothing re-runs."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(
      secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
  )
  runner, session_service = _resumable_runner(
      bank, plugin, [_text_response(), _text_response()]
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  for event in [
      _transfer_event(fc_id, invocation_id="old-inv"),
      _transfer_result_event(fc_id),
  ]:
    await session_service.append_event(session, mark_restored(event))

  await _run(runner, session.id, _user_message())
  await _run(runner, session.id, _user_message("again"))

  assert not bank.transfers


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.asyncio
async def test_e2e_history_copied_in_code_needs_prepare_restored_event(prepare):
  """Copying events into a new session keeps the history only when prepared."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(
      bank,
      plugin,
      [_transfer_response(), _text_response("done"), _text_response("ok")],
  )
  source = await session_service.create_session(app_name=_APP, user_id=_USER)
  await _run(runner, source.id, _user_message("pay bob"))
  assert len(bank.transfers) == 1
  target = await session_service.create_session(app_name=_APP, user_id=_USER)
  for event in _stored_events(session_service, source.id):
    if prepare:
      event = ToolCallIntegrityPlugin.prepare_restored_event(event)
    await session_service.append_event(target, event.model_copy(deep=True))

  if not prepare:
    # The copied stamps are bound to the source session.
    with pytest.raises(RuntimeError) as raised:
      await _run(runner, target.id, _user_message("hi"))
    _assert_rejected_by_plugin(
        raised.value, "before_run_callback", "does not match"
    )
    return

  await _run(runner, target.id, _user_message("hi"))
  assert len(bank.transfers) == 1


@pytest.mark.parametrize(
    "fc_id, match", [("r-1", "restored"), (None, "no ID")], ids=["id", "no_id"]
)
@pytest.mark.parametrize("allow_unstamped_calls", [False, True])
@pytest.mark.asyncio
async def test_e2e_restored_trailing_call_is_blocked_at_execution(
    fc_id, match, allow_unstamped_calls
):
  """History passes before_run; the replayed call is refused before it runs."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(
      secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
  )
  runner, session_service = _resumable_runner(bank, plugin)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, mark_restored(_transfer_event(fc_id))
  )

  with pytest.raises(RuntimeError) as raised:
    await _resume(runner, session.id)

  _assert_rejected_by_plugin(raised.value, "before_tool_callback", match)
  assert not bank.transfers


@pytest.mark.asyncio
async def test_e2e_new_call_after_restore_runs():
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(
      bank, plugin, [_transfer_response(), _text_response("done")]
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, mark_restored(_transfer_event("r-1", invocation_id="old-inv"))
  )

  await _run(runner, session.id, _user_message("pay bob"))

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_restored_copy_does_not_block_a_stamped_call():
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(bank, plugin)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, mark_restored(_transfer_event(invocation_id="old-inv"))
  )
  await session_service.append_event(
      session, await _stamped(plugin, session, _transfer_event())
  )

  await _resume(runner, session.id)

  assert bank.transfers == [{"amount": 1, "recipient": "alice", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_provider_reused_id_after_restore_runs():
  """A model that reuses call IDs across turns can still call tools."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  reused = _transfer_response()
  reused.content.parts[0].function_call.id = "0"
  runner, session_service = _resumable_runner(
      bank, plugin, [reused, _text_response("done")]
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, mark_restored(_transfer_event("0", invocation_id="old-inv"))
  )

  await _run(runner, session.id, _user_message("pay bob"))

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


class _CallWithoutIdAgent(BaseAgent):
  """A custom agent that emits a function call without an ID."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="lookup", args={"q": "x"}
                    )
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_e2e_custom_agent_call_without_id_keeps_session_usable():
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(
          name=_APP,
          root_agent=_CallWithoutIdAgent(name="custom_agent"),
          plugins=[ToolCallIntegrityPlugin(secret_key=_KEY)],
      ),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  await _run(runner, session.id, _user_message("first"))
  await _run(runner, session.id, _user_message("second"))

  stored = _stored_events(session_service, session.id)
  assert sum(1 for e in stored if e.get_function_calls()) == 2


@pytest.mark.asyncio
async def test_e2e_untouched_trailing_call_runs_on_resume():
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(bank, plugin)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, await _stamped(plugin, session, _transfer_event())
  )

  await _resume(runner, session.id)

  assert bank.transfers == [{"amount": 1, "recipient": "alice", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_modified_trailing_call_runs_without_plugin():
  """Without the plugin, a modified trailing call runs on resume as modified."""
  bank = _Bank()
  runner, session_service = _resumable_runner(bank, None)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(session, _transfer_event())
  stored = _stored_events(session_service, session.id)[-1]
  stored.content.parts[0].function_call.args["amount"] = 999

  await _resume(runner, session.id)

  assert bank.transfers == [{"amount": 999, "recipient": "alice", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_modified_trailing_call_is_blocked():
  """A modified trailing call is rejected on resume and does not run."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(bank, plugin)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, await _stamped(plugin, session, _transfer_event())
  )
  stored = _stored_events(session_service, session.id)[-1]
  stored.content.parts[0].function_call.args["amount"] = 999

  with pytest.raises(RuntimeError) as raised:
    await _resume(runner, session.id)

  _assert_rejected_by_plugin(
      raised.value, "before_run_callback", "does not match"
  )
  assert not bank.transfers


@pytest.mark.asyncio
async def test_e2e_duplicated_trailing_call_does_not_run():
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(bank, plugin)
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session, await _stamped(plugin, session, _transfer_event())
  )
  stored = _stored_events(session_service, session.id)[-1]
  stored.content.parts.append(stored.content.parts[0].model_copy(deep=True))

  with pytest.raises(RuntimeError) as raised:
    await _resume(runner, session.id)

  _assert_rejected_by_plugin(raised.value, "before_run_callback", "duplicate")
  assert not bank.transfers


def _duplicate_id_response(other_amount: int) -> LlmResponse:
  response = _transfer_response()
  second = _transfer_response(amount=other_amount).content.parts[0]
  response.content.parts.append(second)
  for part in response.content.parts:
    part.function_call.id = "dup"
  return response


@pytest.mark.parametrize("mode", [StreamingMode.NONE, StreamingMode.SSE])
@pytest.mark.parametrize("other_amount", [2, 999], ids=["same", "different"])
@pytest.mark.parametrize("allow_unstamped_calls", [False, True])
@pytest.mark.asyncio
async def test_e2e_response_repeating_a_call_id_is_refused_and_session_works(
    allow_unstamped_calls, other_amount, mode
):
  """No call from the response runs, and the next turn works."""
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(
      secret_key=_KEY, allow_unstamped_calls=allow_unstamped_calls
  )
  runner, session_service = _resumable_runner(
      bank,
      plugin,
      [
          _duplicate_id_response(other_amount),
          _transfer_response(amount=3),
          _text_response("done"),
      ],
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  async def run(text: str) -> None:
    async for _ in runner.run_async(
        user_id=_USER,
        session_id=session.id,
        new_message=_user_message(text),
        run_config=RunConfig(streaming_mode=mode),
    ):
      pass

  with pytest.raises(RuntimeError) as raised:
    await run("pay bob")

  _assert_rejected_by_plugin(
      raised.value, "on_event_callback", "repeats function call"
  )
  assert not bank.transfers
  stored = _stored_events(session_service, session.id)
  assert not any(e.get_function_calls() for e in stored)

  await run("pay bob 3")

  assert bank.transfers == [{"amount": 3, "recipient": "bob", "memo": None}]


# Fresh calls get an ID before any tool runs, so they pass the gate in every
# execution path.


@pytest.mark.parametrize("mode", [StreamingMode.NONE, StreamingMode.SSE])
@pytest.mark.asyncio
async def test_e2e_fresh_call_runs(mode):
  bank = _Bank()
  plugin = ToolCallIntegrityPlugin(secret_key=_KEY)
  runner, session_service = _resumable_runner(
      bank, plugin, [_transfer_response(), _text_response("done")]
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  async for _ in runner.run_async(
      user_id=_USER,
      session_id=session.id,
      new_message=_user_message("pay bob"),
      run_config=RunConfig(streaming_mode=mode),
  ):
    pass

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_fresh_call_runs_in_workflow_node():
  bank = _Bank()
  agent = LlmAgent(
      name=_AGENT,
      model=_ScriptedModel(
          responses=[_transfer_response(), _text_response("done")]
      ),
      tools=[FunctionTool(bank.transfer_money)],
  )
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(
          name=_APP,
          root_agent=Workflow(name="wf", edges=[(START, agent)]),
          plugins=[ToolCallIntegrityPlugin(secret_key=_KEY)],
      ),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  await _run(runner, session.id, _user_message("pay bob"))

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_tool_as_workflow_node_runs():
  """A tool node has no stored call, so the plugin lets it run."""
  bank = _Bank()
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(
          name=_APP,
          root_agent=Workflow(
              name="wf", edges=[(START, FunctionTool(bank.transfer_money))]
          ),
          plugins=[ToolCallIntegrityPlugin(secret_key=_KEY)],
      ),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  await _run(
      runner, session.id, _user_message('{"amount": 2, "recipient": "bob"}')
  )

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


@pytest.mark.asyncio
async def test_e2e_fresh_call_runs_through_agent_tool():
  bank = _Bank()
  inner = LlmAgent(
      name="inner_agent",
      description="Moves money.",
      model=_ScriptedModel(
          responses=[_transfer_response(), _text_response("moved")]
      ),
      tools=[FunctionTool(bank.transfer_money)],
  )
  outer_call = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="inner_agent", args={"request": "pay bob"}
                  )
              )
          ],
      )
  )
  outer = LlmAgent(
      name="outer_agent",
      model=_ScriptedModel(responses=[outer_call, _text_response("done")]),
      tools=[AgentTool(agent=inner)],
  )
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(
          name=_APP,
          root_agent=outer,
          plugins=[ToolCallIntegrityPlugin(secret_key=_KEY)],
      ),
      session_service=session_service,
  )
  session = await session_service.create_session(app_name=_APP, user_id=_USER)

  await _run(runner, session.id, _user_message("pay bob"))

  assert bank.transfers == [{"amount": 2, "recipient": "bob", "memo": None}]


def test_exported_from_plugins_package():
  from google.adk import plugins  # pylint: disable=g-import-not-at-top

  assert plugins.ToolCallIntegrityPlugin is ToolCallIntegrityPlugin
  assert plugins.ToolCallIntegrityError is ToolCallIntegrityError
