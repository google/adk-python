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

"""Unit tests for the PluginManager."""

from __future__ import annotations

import asyncio
import copy
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.agents import CallbackInfo
from google.adk.agents import CallbackType
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import LIVE_USAGE_KEY
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins._tool_call_integrity_plugin import _HMAC_META_KEY
from google.adk.plugins._tool_call_integrity_plugin import ToolCallIntegrityPlugin
from google.adk.plugins.base_plugin import BasePlugin
# Assume the following path to your modules
# You might need to adjust this based on your project structure.
from google.adk.plugins.plugin_manager import PluginCallbackName
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.runners import Runner
from google.genai import types
from pydantic import ConfigDict
import pytest

from .. import testing_utils


# A helper class to use in tests instead of mocks.
# This makes tests more explicit and easier to debug.
class TestPlugin(BasePlugin):
  __test__ = False
  """
  A test plugin that can be configured to return specific values or raise
  exceptions for any callback, and it logs which callbacks were invoked.
  """

  def __init__(self, name: str):
    super().__init__(name)
    # A log to track the names of callbacks that have been called.
    self.call_log: list[PluginCallbackName] = []
    # A map to configure return values for specific callbacks.
    self.return_values: dict[PluginCallbackName, any] = {}
    # A map to configure exceptions to be raised by specific callbacks.
    self.exceptions_to_raise: dict[PluginCallbackName, Exception] = {}
    self.callback_info_log: list[
        tuple[PluginCallbackName, CallbackInfo | None]
    ] = []

  async def _handle_callback(self, name: PluginCallbackName, **kwargs):
    """Generic handler for all callback methods."""
    self.call_log.append(name)
    callback_context = kwargs.get("callback_context")
    if callback_context is None:
      callback_context = kwargs.get("tool_context")
    if callback_context is not None:
      self.callback_info_log.append((name, callback_context.callback_info))
    if name in self.exceptions_to_raise:
      raise self.exceptions_to_raise[name]
    return self.return_values.get(name)

  # Implement all callback methods from the BasePlugin interface.
  async def on_user_message_callback(self, **kwargs):
    return await self._handle_callback("on_user_message_callback", **kwargs)

  async def before_run_callback(self, **kwargs):
    return await self._handle_callback("before_run_callback", **kwargs)

  async def after_run_callback(self, **kwargs):
    return await self._handle_callback("after_run_callback", **kwargs)

  async def on_event_callback(self, **kwargs):
    return await self._handle_callback("on_event_callback", **kwargs)

  async def before_agent_callback(self, **kwargs):
    return await self._handle_callback("before_agent_callback", **kwargs)

  async def after_agent_callback(self, **kwargs):
    return await self._handle_callback("after_agent_callback", **kwargs)

  async def before_tool_callback(self, **kwargs):
    return await self._handle_callback("before_tool_callback", **kwargs)

  async def after_tool_callback(self, **kwargs):
    return await self._handle_callback("after_tool_callback", **kwargs)

  async def on_tool_error_callback(self, **kwargs):
    return await self._handle_callback("on_tool_error_callback", **kwargs)

  async def before_model_callback(self, **kwargs):
    return await self._handle_callback("before_model_callback", **kwargs)

  async def after_model_callback(self, **kwargs):
    return await self._handle_callback("after_model_callback", **kwargs)

  async def on_model_error_callback(self, **kwargs):
    return await self._handle_callback("on_model_error_callback", **kwargs)

  async def on_agent_error_callback(self, **kwargs):
    return await self._handle_callback("on_agent_error_callback", **kwargs)

  async def on_run_error_callback(self, **kwargs):
    return await self._handle_callback("on_run_error_callback", **kwargs)


@pytest.fixture
def service() -> PluginManager:
  """Provides a clean PluginManager instance for each test."""
  return PluginManager()


@pytest.fixture
def plugin1() -> TestPlugin:
  """Provides a clean instance of our test plugin named 'plugin1'."""
  return TestPlugin(name="plugin1")


@pytest.fixture
def plugin2() -> TestPlugin:
  """Provides a clean instance of our test plugin named 'plugin2'."""
  return TestPlugin(name="plugin2")


def test_register_and_get_plugin(service: PluginManager, plugin1: TestPlugin):
  """Tests successful registration and retrieval of a plugin."""
  service.register_plugin(plugin1)

  assert len(service.plugins) == 1
  assert service.plugins[0] is plugin1
  assert service.get_plugin("plugin1") is plugin1


def test_register_duplicate_plugin_name_raises_value_error(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that registering a plugin with a duplicate name raises an error."""
  plugin1_duplicate = TestPlugin(name="plugin1")
  service.register_plugin(plugin1)

  with pytest.raises(
      ValueError, match="Plugin with name 'plugin1' already registered."
  ):
    service.register_plugin(plugin1_duplicate)


@pytest.mark.asyncio
async def test_early_exit_stops_subsequent_plugins(
    service: PluginManager, plugin1: TestPlugin, plugin2: TestPlugin
):
  """Tests the core "early exit" logic: if a plugin returns a value,

  subsequent plugins for that callback should not be executed.
  """
  # Configure plugin1 to return a value, simulating a cache hit.
  mock_response = Mock(spec=LlmResponse)
  plugin1.return_values["before_run_callback"] = mock_response

  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  # Execute the callback chain.
  result = await service.run_before_run_callback(invocation_context=Mock())

  # Assert that the final result is the one returned by the first plugin.
  assert result is mock_response
  # Assert that the first plugin was called.
  assert "before_run_callback" in plugin1.call_log
  # CRITICAL: Assert that the second plugin was never called.
  assert "before_run_callback" not in plugin2.call_log


@pytest.mark.asyncio
async def test_normal_flow_all_plugins_are_called(
    service: PluginManager, plugin1: TestPlugin, plugin2: TestPlugin
):
  """Tests that if no plugin returns a value, all plugins in the chain

  are executed in order.
  """
  # By default, plugins are configured to return None.
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  result = await service.run_before_run_callback(invocation_context=Mock())

  # The final result should be None as no plugin interrupted the flow.
  assert result is None
  # Both plugins must have been called.
  assert "before_run_callback" in plugin1.call_log
  assert "before_run_callback" in plugin2.call_log


@pytest.mark.parametrize(
    "callback_name, target_name, context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
@pytest.mark.parametrize("edit", ["none", "nonmapping", "nested"])
async def test_each_plugin_sees_owned_metadata_after_edits(
    service, plugin1, plugin2, callback_name, target_name, context_name, edit
):
  """Metadata edits cannot hide original identity from the next plugin."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  metadata = {LIVE_USAGE_KEY: dict(marker)}
  response = (
      Event(author="test_agent", custom_metadata=metadata)
      if target_name == "event"
      else LlmResponse(custom_metadata=metadata)
  )
  observed = []

  async def edit_metadata(**kwargs):
    target = kwargs[target_name]
    if edit == "none":
      target.custom_metadata = None
    elif edit == "nonmapping":
      target.custom_metadata = "not a mapping"
    else:
      target.custom_metadata[LIVE_USAGE_KEY]["sequence"] = 99
      target.custom_metadata["user_tag"] = "voice"
      target.custom_metadata[INTERNAL_METADATA_PREFIX + "plugin_owned"] = True

  async def observe_metadata(**kwargs):
    target = kwargs[target_name]
    observed.append(copy.deepcopy(target.custom_metadata))
    target.custom_metadata["observer_tag"] = "kept"

  setattr(plugin1, callback_name, edit_metadata)
  setattr(plugin2, callback_name, observe_metadata)
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  result = await getattr(service, f"run_{callback_name}")(
      **{target_name: response, context_name: Mock()}
  )

  expected = {LIVE_USAGE_KEY: marker}
  if edit == "nested":
    expected.update({
        "user_tag": "voice",
        INTERNAL_METADATA_PREFIX + "plugin_owned": True,
    })
  assert result is None
  assert observed == [expected]
  assert response.custom_metadata == {**expected, "observer_tag": "kept"}


@pytest.mark.parametrize(
    "callback_name, target_name, context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
@pytest.mark.parametrize("raises", [False, True])
async def test_readonly_callback_lookalikes_keep_results_and_errors(
    service, plugin1, callback_name, target_name, context_name, raises
):
  """Metadata protection must not write to compatible read-only objects."""
  marker = {"connection_id": "connection-1", "sequence": 1}

  class ReadonlyResponse:

    @property
    def custom_metadata(self):
      return {LIVE_USAGE_KEY: marker}

  response = ReadonlyResponse()
  original_error = ValueError("callback failed")
  if raises:
    plugin1.exceptions_to_raise[callback_name] = original_error
  service.register_plugin(plugin1)
  kwargs = {target_name: response, context_name: Mock()}

  if raises:
    with pytest.raises(RuntimeError, match="callback failed") as excinfo:
      await getattr(service, f"run_{callback_name}")(**kwargs)
    assert excinfo.value.__cause__ is original_error
  else:
    result = await getattr(service, f"run_{callback_name}")(**kwargs)
    assert result is None

  assert plugin1.call_log == [callback_name]
  assert response.custom_metadata == {LIVE_USAGE_KEY: marker}


@pytest.mark.parametrize(
    "callback_name, target_name, context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
async def test_unchanged_frozen_event_passes_through_plugins(
    service, plugin1, plugin2, callback_name, target_name, context_name
):
  """No-op callbacks must not trigger assignments on a frozen event."""

  class FrozenEvent(Event):
    model_config = ConfigDict(frozen=True)

  marker = {"connection_id": "connection-1", "sequence": 1}
  event = FrozenEvent(
      id="frozen-event",
      author="test_agent",
      custom_metadata={LIVE_USAGE_KEY: marker},
  )
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  result = await getattr(service, f"run_{callback_name}")(
      **{target_name: event, context_name: Mock()}
  )

  assert result is None
  assert event.custom_metadata == {LIVE_USAGE_KEY: marker}
  assert plugin1.call_log == [callback_name]
  assert plugin2.call_log == [callback_name]


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("edit", ["nested", "clear", "delete"])
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize(
    "callback_name,target_name,context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
async def test_callbacks_restore_provenance_without_reverting_plugin_keys(
    service,
    plugin1,
    plugin2,
    callback_name,
    target_name,
    context_name,
    frozen,
    edit,
    raises,
):
  """Only provenance is restored, including on frozen mutable metadata."""
  response_type = Event if target_name == "event" else LlmResponse
  if frozen:

    class FrozenResponse(response_type):
      model_config = ConfigDict(frozen=True)

    response_type = FrozenResponse
  marker = {"connection_id": "connection-1", "sequence": 1}
  plugin_key = INTERNAL_METADATA_PREFIX + "plugin_owned"
  response = response_type(
      **(
          {"id": "frozen-event", "author": "agent"}
          if target_name == "event"
          else {}
      ),
      custom_metadata={
          LIVE_USAGE_KEY: dict(marker),
          RESTORED_EVENT_KEY: True,
          plugin_key: "old",
      },
  )
  original_error = ValueError("callback failed")
  observed = []

  async def mutate(**kwargs):
    metadata = kwargs[target_name].custom_metadata
    if edit == "nested":
      metadata[LIVE_USAGE_KEY]["sequence"] = 99
      metadata[RESTORED_EVENT_KEY] = False
    elif edit == "clear":
      metadata.clear()
    else:
      del metadata[LIVE_USAGE_KEY]
      del metadata[RESTORED_EVENT_KEY]
    metadata[plugin_key] = "new"
    metadata["user_tag"] = "voice"
    if raises:
      raise original_error

  async def observe(**kwargs):
    observed.append(copy.deepcopy(kwargs[target_name].custom_metadata))

  setattr(plugin1, callback_name, mutate)
  setattr(plugin2, callback_name, observe)
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)
  kwargs = {target_name: response, context_name: Mock()}
  if raises:
    with pytest.raises(RuntimeError, match="callback failed") as excinfo:
      await getattr(service, f"run_{callback_name}")(**kwargs)
    assert excinfo.value.__cause__ is original_error
    assert observed == []
  else:
    await getattr(service, f"run_{callback_name}")(**kwargs)
    assert observed == [response.custom_metadata]
  assert response.custom_metadata == {
      LIVE_USAGE_KEY: marker,
      RESTORED_EVENT_KEY: True,
      plugin_key: "new",
      "user_tag": "voice",
  }


@pytest.mark.parametrize("run_metadata", [None, {"run_tag": "run"}])
@pytest.mark.parametrize("provenance", ["none", "live", "restored", "hmac"])
@pytest.mark.parametrize(
    "replacement_metadata", ["unset", None, {"replacement": True}]
)
async def test_partial_event_replacement_preserves_metadata_contract(
    run_metadata,
    provenance,
    replacement_metadata,
):
  """Unset metadata inherits public tags; explicitly supplied metadata replaces them."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  reserved = {
      "none": {},
      "live": {LIVE_USAGE_KEY: marker},
      "restored": {RESTORED_EVENT_KEY: True},
      "hmac": {_HMAC_META_KEY: {"fc-1": "existing-signature"}},
  }[provenance]
  replacement = Event(
      author="agent", actions=EventActions(state_delta={"seen": True})
  )
  if replacement_metadata != "unset":
    replacement.custom_metadata = replacement_metadata
  original_replacement = copy.deepcopy(replacement)

  class EditAndReplace(BasePlugin):

    async def on_event_callback(self, *, invocation_context, event):
      event.custom_metadata["earlier_tag"] = "kept"
      return replacement

  agent = testing_utils.Agent(name="agent")
  context = await testing_utils.create_invocation_context(
      agent=agent,
      plugins=[EditAndReplace(name="replace")],
      run_config=RunConfig(custom_metadata=run_metadata),
  )
  runner = Runner(
      app=App(name="app", root_agent=agent),
      session_service=context.session_service,
  )
  event = Event(
      author="agent",
      custom_metadata={**reserved, "user_tag": "voice"},
  )
  result = await runner._process_event_with_plugin_callbacks(
      invocation_context=context, event=event
  )

  expected = {**reserved, **(run_metadata or {})}
  if replacement_metadata == "unset":
    expected.update(user_tag="voice", earlier_tag="kept")
  elif replacement_metadata is not None:
    expected.update(replacement_metadata)
  if not reserved and run_metadata is None and replacement_metadata is None:
    expected = None
  assert result.custom_metadata == expected
  assert result.actions.state_delta == {"seen": True}
  if provenance in {"live", "restored"} or replacement_metadata == "unset":
    assert replacement == original_replacement
    assert replacement.model_fields_set == original_replacement.model_fields_set


@pytest.mark.parametrize("live_usage", [False, True])
async def test_integrity_plugin_refresh_survives_event_callbacks(live_usage):
  """A refreshed tool-call signature remains valid in the stored session."""
  integrity = ToolCallIntegrityPlugin(secret_key=b"0" * 32)
  agent = testing_utils.Agent(name="voice")
  context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[integrity]
  )
  runner = Runner(
      app=App(name="app", root_agent=agent),
      session_service=context.session_service,
  )
  metadata = {_HMAC_META_KEY: {"fc-1": "v1:stale"}}
  if live_usage:
    metadata[LIVE_USAGE_KEY] = {"connection_id": "connection-1", "sequence": 1}
  event = Event(
      author="voice",
      custom_metadata=metadata,
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="fc-1", name="lookup", args={"q": "x"}
                  )
              )
          ],
      ),
  )

  result = await runner._process_event_with_plugin_callbacks(
      invocation_context=context, event=event
  )
  await context.session_service.append_event(
      session=context.session, event=result
  )
  await integrity.before_run_callback(invocation_context=context)

  assert result.custom_metadata[_HMAC_META_KEY]["fc-1"] != "v1:stale"


@pytest.mark.parametrize(
    "callback_name, target_name, context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
async def test_nonmapping_metadata_does_not_mask_callback_error(
    service, plugin1, plugin2, callback_name, target_name, context_name
):
  """Restore identity after invalid metadata without masking the exception."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  metadata = {LIVE_USAGE_KEY: marker}
  response = (
      Event(author="test_agent", custom_metadata=metadata)
      if target_name == "event"
      else LlmResponse(custom_metadata=metadata)
  )
  original_error = ValueError("callback failed after metadata edit")

  async def edit_and_raise(**kwargs):
    kwargs[target_name].custom_metadata = "not a mapping"
    raise original_error

  setattr(plugin1, callback_name, edit_and_raise)
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  with pytest.raises(RuntimeError, match="callback failed") as excinfo:
    await getattr(service, f"run_{callback_name}")(
        **{target_name: response, context_name: Mock()}
    )

  assert excinfo.value.__cause__ is original_error
  assert response.custom_metadata == {LIVE_USAGE_KEY: marker}
  assert plugin2.call_log == []


@pytest.mark.parametrize(
    "callback_name, target_name, context_name",
    [
        ("after_model_callback", "llm_response", "callback_context"),
        ("on_event_callback", "event", "invocation_context"),
    ],
)
async def test_replacement_keeps_owned_metadata_without_mutation(
    service, plugin1, plugin2, callback_name, target_name, context_name
):
  """An early replacement is copied, preserving identity and early exit."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  original_metadata = {LIVE_USAGE_KEY: dict(marker)}
  replacement_metadata = {
      "user_tag": "replacement",
      LIVE_USAGE_KEY: {"connection_id": "different", "sequence": 99},
      INTERNAL_METADATA_PREFIX + "plugin_owned": True,
  }
  if target_name == "event":
    original = Event(author="test_agent", custom_metadata=original_metadata)
    replacement = Event(
        author="test_agent", custom_metadata=replacement_metadata
    )
  else:
    original = LlmResponse(custom_metadata=original_metadata)
    replacement = LlmResponse(custom_metadata=replacement_metadata)
  saved_replacement = replacement.model_dump()
  plugin1.return_values[callback_name] = replacement
  service.register_plugin(plugin1)
  service.register_plugin(plugin2)

  result = await getattr(service, f"run_{callback_name}")(
      **{target_name: original, context_name: Mock()}
  )

  assert result is not replacement
  assert result.custom_metadata == {
      **replacement_metadata,
      LIVE_USAGE_KEY: marker,
  }
  assert replacement.model_dump() == saved_replacement
  assert original.custom_metadata == original_metadata
  assert plugin1.call_log == [callback_name]
  assert plugin2.call_log == []


@pytest.mark.asyncio
async def test_plugin_exception_is_wrapped_in_runtime_error(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that if a plugin callback raises an exception, the PluginManager

  catches it and raises a descriptive RuntimeError.
  """
  # Configure the plugin to raise an error during a specific callback.
  original_exception = ValueError("Something went wrong inside the plugin!")
  plugin1.exceptions_to_raise["before_run_callback"] = original_exception
  service.register_plugin(plugin1)

  with pytest.raises(RuntimeError) as excinfo:
    await service.run_before_run_callback(invocation_context=Mock())

  # Check that the error message is informative.
  assert "Error in plugin 'plugin1'" in str(excinfo.value)
  assert "before_run_callback" in str(excinfo.value)
  # Check that the original exception is chained for better tracebacks.
  assert excinfo.value.__cause__ is original_exception


@pytest.mark.asyncio
async def test_all_callbacks_are_supported(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that all callbacks defined in the BasePlugin interface are supported

  by the PluginManager.
  """
  service.register_plugin(plugin1)
  mock_context = Mock()
  mock_user_message = Mock()
  mock_invocation_context = Mock()
  mock_invocation_context.session.state = {}
  mock_invocation_context._state_schema = None
  callback_context = CallbackContext(mock_invocation_context)

  # Test all callbacks
  await service.run_on_user_message_callback(
      user_message=mock_user_message, invocation_context=mock_context
  )
  await service.run_before_run_callback(invocation_context=mock_context)
  await service.run_after_run_callback(invocation_context=mock_context)
  await service.run_on_event_callback(
      invocation_context=mock_context, event=mock_context
  )
  await service.run_before_agent_callback(
      agent=mock_context, callback_context=callback_context
  )
  await service.run_after_agent_callback(
      agent=mock_context, callback_context=callback_context
  )
  await service.run_before_tool_callback(
      tool=mock_context, tool_args={}, tool_context=callback_context
  )
  await service.run_after_tool_callback(
      tool=mock_context, tool_args={}, tool_context=callback_context, result={}
  )
  await service.run_on_tool_error_callback(
      tool=mock_context,
      tool_args={},
      tool_context=callback_context,
      error=mock_context,
  )
  await service.run_before_model_callback(
      callback_context=callback_context, llm_request=mock_context
  )
  await service.run_after_model_callback(
      callback_context=callback_context, llm_response=mock_context
  )
  await service.run_on_model_error_callback(
      callback_context=callback_context,
      llm_request=mock_context,
      error=mock_context,
  )
  await service.run_on_agent_error_callback(
      agent=mock_context,
      callback_context=callback_context,
      error=mock_context,
  )
  await service.run_on_run_error_callback(
      invocation_context=mock_context,
      error=mock_context,
  )

  # Verify all callbacks were logged
  expected_callbacks = [
      "on_user_message_callback",
      "before_run_callback",
      "after_run_callback",
      "on_event_callback",
      "before_agent_callback",
      "after_agent_callback",
      "before_tool_callback",
      "after_tool_callback",
      "on_tool_error_callback",
      "before_model_callback",
      "after_model_callback",
      "on_model_error_callback",
      "on_agent_error_callback",
      "on_run_error_callback",
  ]
  assert set(plugin1.call_log) == set(expected_callbacks)
  assert plugin1.callback_info_log == [
      (
          f"{callback_type.value}_callback",
          CallbackInfo(type=callback_type),
      )
      for callback_type in (
          CallbackType.BEFORE_AGENT,
          CallbackType.AFTER_AGENT,
          CallbackType.BEFORE_TOOL,
          CallbackType.AFTER_TOOL,
          CallbackType.ON_TOOL_ERROR,
          CallbackType.BEFORE_MODEL,
          CallbackType.AFTER_MODEL,
          CallbackType.ON_MODEL_ERROR,
          CallbackType.ON_AGENT_ERROR,
      )
  ]


@pytest.mark.asyncio
async def test_plugin_callback_receives_current_callback_info():
  """Callback info names the active plugin callback type only while it runs."""

  class _RecordingPlugin(BasePlugin):

    def __init__(self) -> None:
      super().__init__(name="recording_plugin")
      self.callback_info: CallbackInfo | None = None

    async def before_model_callback(
        self, *, callback_context, llm_request
    ) -> None:
      self.callback_info = callback_context.callback_info

  invocation_context = Mock()
  invocation_context.session.state = {}
  invocation_context._state_schema = None
  callback_context = CallbackContext(invocation_context)
  plugin = _RecordingPlugin()
  service = PluginManager(plugins=[plugin])

  assert callback_context.callback_info is None

  await service.run_before_model_callback(
      callback_context=callback_context,
      llm_request=Mock(),
  )

  assert plugin.callback_info == CallbackInfo(type=CallbackType.BEFORE_MODEL)
  assert callback_context.callback_info is None


@pytest.mark.asyncio
async def test_concurrent_plugin_callbacks_keep_callback_info_isolated():
  """Concurrent callbacks with separate contexts see only their own type."""
  model_started = asyncio.Event()
  tool_started = asyncio.Event()
  model_observed = asyncio.Event()
  observations = []

  class _ConcurrentPlugin(BasePlugin):

    async def before_model_callback(
        self, *, callback_context, llm_request
    ) -> None:
      model_started.set()
      await tool_started.wait()
      observations.append(callback_context.callback_info.type)
      model_observed.set()

    async def before_tool_callback(
        self, *, tool, tool_args, tool_context
    ) -> None:
      await model_started.wait()
      tool_started.set()
      await model_observed.wait()
      observations.append(tool_context.callback_info.type)

  invocation_context = Mock()
  invocation_context.session.state = {}
  invocation_context._state_schema = None
  model_callback_context = CallbackContext(invocation_context)
  tool_callback_context = CallbackContext(invocation_context)
  service = PluginManager(plugins=[_ConcurrentPlugin("concurrent_plugin")])

  await asyncio.gather(
      service.run_before_model_callback(
          callback_context=model_callback_context,
          llm_request=Mock(),
      ),
      service.run_before_tool_callback(
          tool=Mock(),
          tool_args={},
          tool_context=tool_callback_context,
      ),
  )

  assert observations == [
      CallbackType.BEFORE_MODEL,
      CallbackType.BEFORE_TOOL,
  ]
  assert model_callback_context.callback_info is None
  assert tool_callback_context.callback_info is None


@pytest.mark.asyncio
async def test_unmapped_plugin_callbacks_fall_back_to_nullcontext():
  """Unmapped plugin callback names fall back to nullcontext()."""
  observed_infos: list[CallbackInfo | None] = []

  class _CustomPlugin(BasePlugin):

    async def custom_callback(self, *, callback_context) -> None:
      observed_infos.append(callback_context.callback_info)

    async def custom_notification_callback(self, *, callback_context) -> None:
      observed_infos.append(callback_context.callback_info)

  invocation_context = Mock()
  invocation_context.session.state = {}
  invocation_context._state_schema = None
  callback_context = CallbackContext(invocation_context)
  service = PluginManager(plugins=[_CustomPlugin("custom_plugin")])

  await service._run_callbacks(
      "custom_callback",  # type: ignore[arg-type]
      callback_context=callback_context,
  )
  await service._run_notification_callbacks(
      "custom_notification_callback",  # type: ignore[arg-type]
      callback_context=callback_context,
  )

  assert observed_infos == [None, None]
  assert callback_context.callback_info is None


@pytest.mark.asyncio
async def test_close_calls_plugin_close(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that close calls the close method on registered plugins."""
  plugin1.close = AsyncMock()
  service.register_plugin(plugin1)

  await service.close()

  plugin1.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_raises_runtime_error_on_plugin_exception(
    service: PluginManager, plugin1: TestPlugin
):
  """Tests that close raises a RuntimeError if a plugin's close fails."""
  plugin1.close = AsyncMock(side_effect=ValueError("Shutdown error"))
  service.register_plugin(plugin1)

  with pytest.raises(
      RuntimeError, match="Failed to close plugins: 'plugin1': ValueError"
  ):
    await service.close()

  plugin1.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_closes_remaining_plugins_and_reraises_cancellation():
  """Tests that cancellation propagates only after all plugins are closed."""
  plugins = [TestPlugin(name=f"plugin{i}") for i in range(1, 4)]
  for plugin in plugins:
    plugin.close = AsyncMock()
  plugins[1].close = AsyncMock(side_effect=asyncio.CancelledError())
  service = PluginManager(plugins=plugins)

  with pytest.raises(asyncio.CancelledError):
    await service.close()

  for plugin in plugins:
    plugin.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_closes_remaining_plugins_when_task_is_cancelled():
  """Tests that a real task cancellation still closes the remaining plugins."""
  started = asyncio.Event()

  async def blocking_close():
    started.set()
    await asyncio.sleep(10)

  plugins = [TestPlugin(name=f"plugin{i}") for i in range(1, 4)]
  plugins[0].close = AsyncMock()
  plugins[1].close = blocking_close
  plugins[2].close = AsyncMock()
  service = PluginManager(plugins=plugins)
  task = asyncio.create_task(service.close())
  await started.wait()

  task.cancel()

  with pytest.raises(asyncio.CancelledError):
    await task

  plugins[0].close.assert_awaited_once()
  plugins[2].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_closes_remaining_plugins_after_plugin_error():
  """Tests that a failing plugin does not stop the remaining ones closing."""
  plugins = [TestPlugin(name=f"plugin{i}") for i in range(1, 4)]
  for plugin in plugins:
    plugin.close = AsyncMock()
  plugins[1].close = AsyncMock(side_effect=ValueError("Shutdown error"))
  service = PluginManager(plugins=plugins)

  with pytest.raises(RuntimeError, match="'plugin2': ValueError"):
    await service.close()

  for plugin in plugins:
    plugin.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_with_timeout(plugin1: TestPlugin):
  """Tests that close respects the timeout and raises on failure."""
  service = PluginManager(close_timeout=0.1)

  async def slow_close():
    await asyncio.sleep(0.2)

  plugin1.close = slow_close
  service.register_plugin(plugin1)

  with pytest.raises(RuntimeError) as excinfo:
    await service.close()

  assert "Failed to close plugins: 'plugin1': TimeoutError" in str(
      excinfo.value
  )


@pytest.mark.asyncio
async def test_close_is_noop_after_set_skip_closing_plugins(
    plugin1: TestPlugin,
):
  """Tests that close is a no-op after set_skip_closing_plugins(True)."""
  plugin1.close = AsyncMock()
  service = PluginManager(plugins=[plugin1])
  service.set_skip_closing_plugins(True)

  await service.close()

  plugin1.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_skip_closing_plugins_bypasses_close_timeout(
    plugin1: TestPlugin,
):
  """Tests that set_skip_closing_plugins(True) bypasses close timeout."""

  async def slow_close():
    await asyncio.sleep(10)  # Would otherwise time out.

  plugin1.close = slow_close
  service = PluginManager(plugins=[plugin1], close_timeout=0.01)
  service.set_skip_closing_plugins(True)

  # Should return immediately without raising.
  await service.close()


@pytest.mark.asyncio
async def test_set_skip_closing_plugins_false_reverts_to_closing(
    plugin1: TestPlugin,
):
  """Tests that set_skip_closing_plugins(False) re-enables closing."""
  plugin1.close = AsyncMock()
  service = PluginManager(plugins=[plugin1])
  service.set_skip_closing_plugins(True)
  service.set_skip_closing_plugins(False)

  await service.close()

  plugin1.close.assert_awaited_once()
