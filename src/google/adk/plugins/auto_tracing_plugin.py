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

"""Monkey-patches Python fns to emit OpenTelemetry spans."""

from __future__ import annotations

import inspect
import logging
import sys
import threading
from typing import TYPE_CHECKING

from opentelemetry import trace
from typing_extensions import override

from . import auto_tracing_helpers
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_WALK_DEPTH = 30

_ATOMIC_TYPES = (str, bytes, int, float, bool, type(None))


class AutoTracingPlugin(BasePlugin):
  """Auto-instruments in-scope Python functions with OpenTelemetry spans."""

  def __init__(
      self,
      *,
      name: str = "AutoTracingPlugin",
      extra_scope_prefixes: tuple[str, ...] = (),
      tracer: trace.Tracer | None = None,
      max_repr_len: int = auto_tracing_helpers.DEFAULT_MAX_REPR_LEN,
      max_recorded_yields: int = auto_tracing_helpers.DEFAULT_MAX_RECORDED_YIELDS,
      max_walk_depth: int = DEFAULT_MAX_WALK_DEPTH,
  ):
    super().__init__(name=name)
    self._scope_prefixes = tuple(extra_scope_prefixes)
    self._tracer = tracer or trace.get_tracer(__name__)
    self._caps = auto_tracing_helpers.Caps(
        max_repr_len=max_repr_len,
        max_recorded_yields=max_recorded_yields,
    )
    self._max_walk_depth = max_walk_depth
    self._tracer_eligible = auto_tracing_helpers.tracer_will_record(
        self._tracer
    )
    self._lock = threading.Lock()
    self._wrapped_modules: set[str] = set()

  @override
  async def before_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    if not self._tracer_eligible:
      return
    with self._lock:
      self._add_agent_scope(invocation_context)
      for name in list(sys.modules):
        if name in self._wrapped_modules or name == __name__:
          continue
        if not name.startswith(self._scope_prefixes):
          continue
        module = sys.modules.get(name)
        if module is None:
          continue
        try:
          self._wrap_module(module)
        except Exception:  # pylint: disable=broad-exception-caught
          logger.exception("AutoTracingPlugin: failed to instrument %s", name)
        self._wrapped_modules.add(name)

  def _add_agent_scope(self, invocation_context):
    """Adds packages of every object reachable from the invocation."""
    seen, packages = set(), set()
    max_depth = self._max_walk_depth

    def walk(obj, depth):
      if obj is None:
        return
      if depth > max_depth or id(obj) in seen:
        return
      if isinstance(obj, _ATOMIC_TYPES):
        return
      seen.add(id(obj))
      module = getattr(obj, "__module__", None) or getattr(
          getattr(obj, "func", None),
          "__module__",
          None,
      )
      if module:
        # Top-level "mod" needs both "mod" and "mod." to match name.startswith.
        if "." in module:
          packages.add(module.rsplit(".", 1)[0] + ".")
        else:
          packages.add(module)
          packages.add(module + ".")
      if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
          walk(item, depth + 1)
        return
      if isinstance(obj, dict):
        for item in obj.values():
          walk(item, depth + 1)
        return
      # Avoid getattr on instance __dict__ so @property / lazy descriptors don't fire.
      instance_dict = getattr(obj, "__dict__", None)
      if isinstance(instance_dict, dict):
        for attr_name, value in instance_dict.items():
          if not attr_name.startswith("_"):
            walk(value, depth + 1)
      for slot_name in auto_tracing_helpers.public_slot_names(type(obj)):
        try:
          value = getattr(obj, slot_name)
        except AttributeError:
          continue
        walk(value, depth + 1)

    walk(getattr(invocation_context, "agent", None), 0)
    new = tuple(sorted(packages - set(self._scope_prefixes)))
    if new:
      self._scope_prefixes = self._scope_prefixes + new

  def _wrap_module(self, module):
    module_name = module.__name__
    for attr_name, attr in inspect.getmembers(module):
      if attr_name.startswith("_"):
        continue
      if getattr(attr, "__module__", "") != module_name:
        continue
      if inspect.isfunction(attr):
        self._rebind(module, attr_name, attr)
      elif inspect.isclass(attr):
        for member_name, member in inspect.getmembers(attr):
          if member_name.startswith("__"):
            continue
          if not inspect.isfunction(member):
            continue
          if getattr(member, "__module__", "") != module_name:
            continue
          self._rebind(attr, member_name, member)

  def _rebind(self, owner, name, fn):
    if getattr(fn, auto_tracing_helpers.WRAPPED_ATTR, False):
      return
    try:
      setattr(
          owner,
          name,
          auto_tracing_helpers.build_tracing_wrapper(
              fn, self._tracer, self._caps
          ),
      )
    except (AttributeError, TypeError) as exc:
      logger.info(
          "AutoTracingPlugin: cannot rebind %s.%s: %s",
          getattr(owner, "__qualname__", owner),
          name,
          exc,
      )
