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

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest import mock

from google.adk.plugins import auto_tracing_helpers
from google.adk.plugins import auto_tracing_plugin
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import export as trace_export
from opentelemetry.sdk.trace.export import in_memory_span_exporter
import pytest

_FIXTURE_MODULE_NAME = (
    "google.adk.tests.unittests.plugins.synthetic_test_fixture"
)


def _sync_fn(x: int) -> int:
  return x + 1


async def _async_fn(x: int) -> int:
  return x * 2


def _build_fixture_module() -> types.ModuleType:
  module = types.ModuleType(_FIXTURE_MODULE_NAME)
  module.__name__ = _FIXTURE_MODULE_NAME
  for fn in (_sync_fn, _async_fn):
    fn.__module__ = _FIXTURE_MODULE_NAME

  def _method(unused_self, x: int) -> int:
    return x - 1

  async def _async_method(unused_self, x: int) -> int:
    return x + 10

  _method.__module__ = _FIXTURE_MODULE_NAME
  _async_method.__module__ = _FIXTURE_MODULE_NAME
  cls = type("C", (), {"method": _method, "async_method": _async_method})
  cls.__module__ = _FIXTURE_MODULE_NAME

  module.sync_fn = _sync_fn
  module.async_fn = _async_fn
  module.C = cls
  return module


def _install_module(name: str, fn) -> types.ModuleType:
  mod = types.ModuleType(name)
  mod.__name__ = name
  fn.__module__ = name
  mod.fn = fn
  sys.modules[name] = mod
  return mod


def _run_sync(module):
  return module.sync_fn(7)


def _run_async(module):
  return asyncio.run(module.async_fn(4))


def _run_class_method(module):
  return module.C().method(5)


def _run_class_async_method(module):
  return asyncio.run(module.C().async_method(5))


class _Ctx:

  def __init__(self, agent):
    self.agent = agent


def _build_slot_agent(module: str, slots, attr: str, value):
  cls = type("_Agent", (), {"__slots__": slots, "__module__": module})
  obj = cls()
  setattr(obj, attr, value)
  return obj


def _sub_helper():
  return 7


@pytest.fixture
def fixture():
  exporter = in_memory_span_exporter.InMemorySpanExporter()
  provider = trace_sdk.TracerProvider()
  provider.add_span_processor(trace_export.SimpleSpanProcessor(exporter))
  tracer = provider.get_tracer("test")
  module = _build_fixture_module()
  sys.modules[_FIXTURE_MODULE_NAME] = module
  yield types.SimpleNamespace(exporter=exporter, tracer=tracer, module=module)
  sys.modules.pop(_FIXTURE_MODULE_NAME, None)


def _span_names(exporter) -> list[str]:
  return [s.name for s in exporter.get_finished_spans()]


def _attrs_for(exporter, substr: str) -> dict[str, Any]:
  matches = [
      dict(s.attributes or {})
      for s in exporter.get_finished_spans()
      if substr in s.name
  ]
  assert matches, f"no span matched {substr!r} in {_span_names(exporter)}"
  return matches[0]


def _instrument(
    tracer, scope_prefixes=(_FIXTURE_MODULE_NAME,)
) -> auto_tracing_plugin.AutoTracingPlugin:
  plugin = auto_tracing_plugin.AutoTracingPlugin(
      tracer=tracer, extra_scope_prefixes=scope_prefixes
  )
  asyncio.run(plugin.before_run_callback(invocation_context=None))
  return plugin


@pytest.mark.parametrize(
    "run_fn,expected_substr",
    [
        (_run_sync, "_sync_fn"),
        (_run_async, "_async_fn"),
        (_run_class_method, "._method"),
        (_run_class_async_method, "._async_method"),
    ],
)
def test_emits_span(fixture, run_fn, expected_substr):
  _instrument(fixture.tracer)
  run_fn(fixture.module)
  assert any(
      expected_substr in n for n in _span_names(fixture.exporter)
  ), f"missing {expected_substr!r} in {_span_names(fixture.exporter)}"


@pytest.mark.parametrize(
    "run_fn,expected_substr,expected_attrs",
    [
        (
            _run_sync,
            "_sync_fn",
            {"adk.fn.arg.x": "7", "adk.fn.return": "8"},
        ),
        (_run_async, "_async_fn", {"adk.fn.return": "8"}),
    ],
)
def test_records_io(fixture, run_fn, expected_substr, expected_attrs):
  _instrument(fixture.tracer)
  run_fn(fixture.module)
  attrs = _attrs_for(fixture.exporter, expected_substr)
  assert {k: attrs.get(k) for k in expected_attrs} == expected_attrs


@pytest.mark.parametrize("attr", ["sync_fn", "async_fn"])
def test_repeat_instrument_is_idempotent(fixture, attr):
  plugin = _instrument(fixture.tracer)
  first = getattr(fixture.module, attr)
  asyncio.run(plugin.before_run_callback(invocation_context=None))
  assert getattr(fixture.module, attr) is first


@pytest.mark.parametrize("attr", ["sync_fn", "async_fn"])
def test_wrapper_marker_is_true(fixture, attr):
  _instrument(fixture.tracer)
  assert (
      getattr(getattr(fixture.module, attr), auto_tracing_helpers.WRAPPED_ATTR)
      is True
  )


def test_out_of_scope_module_is_not_instrumented(fixture):
  name = "auto_tracing_plugin_test_not_in_scope"
  mod = _install_module(name, lambda: 42)
  try:
    _instrument(fixture.tracer)
    mod.fn()
    assert f"{name}.fn" not in _span_names(fixture.exporter)
  finally:
    sys.modules.pop(name, None)


def test_records_exception(fixture):
  name = "auto_tracing_plugin_test_boom"

  def boom():
    raise ValueError("kaboom")

  mod = _install_module(name, boom)
  try:
    _instrument(fixture.tracer, scope_prefixes=(name,))
    with pytest.raises(ValueError, match="kaboom"):
      mod.fn()
    attrs = _attrs_for(fixture.exporter, "boom")
    assert attrs.get("adk.fn.exc_type") == "ValueError"
    assert "kaboom" in attrs.get("adk.fn.exc_repr", "")
  finally:
    sys.modules.pop(name, None)


def test_walk_returns_quickly_on_none_agent(fixture):
  plugin = auto_tracing_plugin.AutoTracingPlugin(tracer=fixture.tracer)
  asyncio.run(plugin.before_run_callback(invocation_context=_Ctx(None)))
  assert _span_names(fixture.exporter) == []


def test_add_agent_scope_picks_up_agent_package(fixture):
  pkg = "auto_tracing_plugin_test_agent_pkg"
  mod_name = f"{pkg}.helpers"

  def helper():
    return 99

  mod = _install_module(mod_name, helper)
  try:

    class _Agent:
      __module__ = f"{pkg}.agent"

    plugin = auto_tracing_plugin.AutoTracingPlugin(tracer=fixture.tracer)
    asyncio.run(plugin.before_run_callback(invocation_context=_Ctx(_Agent())))
    mod.fn()
    assert any("helper" in n for n in _span_names(fixture.exporter)), (
        f"agent pkg {pkg!r} was not absorbed;"
        f" spans={_span_names(fixture.exporter)}"
    )
  finally:
    sys.modules.pop(mod_name, None)


@pytest.mark.parametrize(
    "pkg,slots",
    [
        ("auto_tracing_plugin_test_slots_pkg", ("child",)),
        ("auto_tracing_plugin_test_str_slot_pkg", "child"),
    ],
)
def test_add_agent_scope_walks_slots_attrs(fixture, pkg, slots):
  sub_mod_name = f"{pkg}.sub"
  mod = _install_module(sub_mod_name, _sub_helper)
  try:
    sub = type("_Sub", (), {"__module__": sub_mod_name})()
    agent = _build_slot_agent(f"{pkg}.agent", slots, "child", sub)
    plugin = auto_tracing_plugin.AutoTracingPlugin(tracer=fixture.tracer)
    asyncio.run(plugin.before_run_callback(invocation_context=_Ctx(agent)))
    mod.fn()
    assert any("_sub_helper" in n for n in _span_names(fixture.exporter)), (
        f"slot-referenced pkg {pkg!r} was not absorbed;"
        f" spans={_span_names(fixture.exporter)}"
    )
  finally:
    sys.modules.pop(sub_mod_name, None)


def test_add_agent_scope_does_not_fire_property_descriptors(fixture):
  fired: list[str] = []

  class _Agent:
    __module__ = "auto_tracing_plugin_test_no_descriptor_pkg.agent"

    @property
    def expensive(self):
      fired.append("expensive")
      raise RuntimeError("should never be invoked during scope walk")

  plugin = auto_tracing_plugin.AutoTracingPlugin(tracer=fixture.tracer)
  asyncio.run(plugin.before_run_callback(invocation_context=_Ctx(_Agent())))
  assert fired == [], f"@property fired during agent-scope walk: {fired!r}"


def test_module_removed_mid_iteration_does_not_log_exception(fixture):
  name = "auto_tracing_plugin_test_disappearing"
  _install_module(name, lambda: 1)
  try:
    plugin = auto_tracing_plugin.AutoTracingPlugin(
        tracer=fixture.tracer, extra_scope_prefixes=(name,)
    )

    class _DroppingModules(dict):

      def get(self, key, default=None):
        if key == name:
          return None
        return super().get(key, default)

    dropping = _DroppingModules(sys.modules)
    with (
        mock.patch.object(
            auto_tracing_plugin.logger, "exception", autospec=True
        ) as log_exc,
        mock.patch.object(auto_tracing_plugin.sys, "modules", new=dropping),
    ):
      asyncio.run(plugin.before_run_callback(invocation_context=None))
    assert (
        not log_exc.called
    ), f"unexpected logger.exception calls: {log_exc.call_args_list}"
    assert name not in plugin._wrapped_modules
  finally:
    sys.modules.pop(name, None)


def test_repeat_instrument_does_not_rewrap(fixture):
  plugin = _instrument(fixture.tracer)
  assert getattr(
      fixture.module.sync_fn, auto_tracing_helpers.WRAPPED_ATTR, False
  )
  assert _FIXTURE_MODULE_NAME in plugin._wrapped_modules
  with mock.patch.object(plugin, "_wrap_module", autospec=True) as wrap_module:
    asyncio.run(plugin.before_run_callback(invocation_context=None))
  wrap_module.assert_not_called()


class _Slotted:
  __slots__ = ("a", "b")

  def __init__(self):
    self.a = 1
    self.b = "x"


class _Bare:
  __slots__ = ()


@pytest.mark.parametrize(
    "instance,expected_substrings",
    [
        (_Slotted(), ("_Slotted", "a=1", "b='x'")),
        (_Bare(), ("<_Bare>",)),
    ],
)
def test_summarize_default(instance, expected_substrings):
  rendered = auto_tracing_helpers.safe_repr(
      instance, auto_tracing_helpers.Caps()
  )
  for s in expected_substrings:
    assert s in rendered, rendered


def test_add_agent_scope_picks_up_top_level_module(fixture):
  top_mod_name = "auto_tracing_plugin_test_top_level_pkg"

  def top_helper():
    return 1

  mod = _install_module(top_mod_name, top_helper)
  try:

    class _Agent:
      __module__ = top_mod_name

    plugin = auto_tracing_plugin.AutoTracingPlugin(tracer=fixture.tracer)
    asyncio.run(plugin.before_run_callback(invocation_context=_Ctx(_Agent())))
    mod.fn()
    assert any("top_helper" in n for n in _span_names(fixture.exporter)), (
        f"top-level module {top_mod_name!r} not absorbed;"
        f" spans={_span_names(fixture.exporter)}"
    )
  finally:
    sys.modules.pop(top_mod_name, None)


def test_signature_introspection_happens_once_per_wrap(fixture):
  with mock.patch.object(
      auto_tracing_helpers.inspect, "signature", autospec=True
  ) as sig:
    sig.side_effect = auto_tracing_helpers.inspect.signature
    _instrument(fixture.tracer)
    wrap_calls = sig.call_count
    for _ in range(5):
      _run_sync(fixture.module)
      _run_async(fixture.module)
  assert (
      sig.call_count == wrap_calls
  ), f"inspect.signature called per-call: {wrap_calls} -> {sig.call_count}"


def test_async_gen_caps_buffered_items(fixture):
  cap = 3
  total_yields = 100
  name = "auto_tracing_plugin_test_async_gen_cap"

  async def producer():
    for i in range(total_yields):
      yield i

  mod = _install_module(name, producer)
  try:
    plugin = auto_tracing_plugin.AutoTracingPlugin(
        tracer=fixture.tracer,
        extra_scope_prefixes=(name,),
        max_recorded_yields=cap,
    )
    asyncio.run(plugin.before_run_callback(invocation_context=None))

    async def drive():
      seen = []
      async for x in mod.fn():
        seen.append(x)
      return seen

    out = asyncio.run(drive())
    assert out == list(range(total_yields))
    attrs = _attrs_for(fixture.exporter, "producer")
    rendered = attrs.get("adk.fn.return", "")
    assert f"{total_yields} items yielded" in rendered, rendered
    assert f"first {cap}:" in rendered, rendered
    assert f"+ {total_yields - cap} more" in rendered, rendered
  finally:
    sys.modules.pop(name, None)


def test_sync_gen_caps_buffered_items(fixture):
  cap = 2
  total_yields = 50
  name = "auto_tracing_plugin_test_sync_gen_cap"

  def producer():
    for i in range(total_yields):
      yield i

  mod = _install_module(name, producer)
  try:
    plugin = auto_tracing_plugin.AutoTracingPlugin(
        tracer=fixture.tracer,
        extra_scope_prefixes=(name,),
        max_recorded_yields=cap,
    )
    asyncio.run(plugin.before_run_callback(invocation_context=None))
    out = list(mod.fn())
    assert out == list(range(total_yields))
    attrs = _attrs_for(fixture.exporter, "producer")
    rendered = attrs.get("adk.fn.return", "")
    assert f"{total_yields} items yielded" in rendered, rendered
    assert f"first {cap}:" in rendered, rendered
  finally:
    sys.modules.pop(name, None)
