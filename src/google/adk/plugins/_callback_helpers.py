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

"""Callback contracts shared by the observability plugins.

The built-in observability plugins (``LoggingPlugin``, ``DebugLoggingPlugin``
and ``BigQueryAgentAnalyticsPlugin``) read the same callbacks, and each must
reach the same verdict on the same inputs:

* A tool result that reports a failure without raising, such as a
  ``ReflectAndRetryToolPlugin`` answer or an MCP ``CallToolResult`` with
  ``isError`` set, is a tool error. An application can classify its own tools'
  results with a ``ToolResultClassifier``.
* A call whose failure ``on_tool_error_callback`` recorded is not recorded
  again when a later plugin's answer to that failure reaches
  ``after_tool_callback`` as the call's result.
* An ``Event.error_code`` that is a model finish or block reason is a model
  termination, not a workflow node failure.

None of the tool result checks run code of the result, its keys or its
values, so a hostile or malformed result cannot raise into a callback or leak
its content into a log.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextvars
from dataclasses import dataclass
import functools
import inspect
import logging
import sys
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING
import weakref

from google.genai import types

from ._reflect_retry_utils import REFLECT_AND_RETRY_RESPONSE_TYPE

if TYPE_CHECKING:
  from ..events.event import Event
  from ..tools.base_tool import BaseTool
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

# The values an ``LlmResponse`` copies into ``Event.error_code`` when the
# model stops or is blocked. They report a model termination, not a node
# failure.
_LLM_RESPONSE_ERROR_CODES = frozenset(
    reason.value
    for reason_type in (types.FinishReason, types.BlockedReason)
    for reason in reason_type
)

_MCP_TOOL_ERROR_MESSAGE = "Tool returned an MCP result with isError=true."
_REFLECT_AND_RETRY_TOOL_ERROR_MESSAGE = (
    "Tool raised an error that ReflectAndRetryToolPlugin handled."
)
_CLASSIFIED_TOOL_ERROR_MESSAGE = (
    "tool_result_classifier classified the tool result as an error."
)
# MCP SDK 1.x defines CallToolResult in mcp.types. 2.x defines it in the
# mcp_types package, which mcp.types re-exports, so a program can hold one
# without having imported mcp.types.
_MCP_TYPES_MODULES = ("mcp.types", "mcp_types")
# The only entries of a tool result that the built-in rules read.
_RESULT_FIELD_NAMES = frozenset(
    {"response_type", "error_details", "error_type", "isError", "is_error"}
)


def _is_durable_node_error(event: Event) -> bool:
  """Returns whether an event reports that a workflow node failed.

  A partial event is not final. An ``error_code`` that is a model finish or
  block reason was copied from an ``LlmResponse`` and reports why the model
  stopped, so it is not a node failure either.
  """
  return (
      event.partial is not True
      and bool(event.error_code)
      and event.error_code not in _LLM_RESPONSE_ERROR_CODES
  )


def _imported_types(
    module_name: str, *type_names: str
) -> tuple[type[object], ...]:
  """Returns the named classes of a module the process has already imported.

  Looking the module up in ``sys.modules`` never imports it: ``toolbox_adk`` is
  an optional dependency, and importing a module the application does not use
  would slow down its first tool call. No tool can be an instance of a class
  whose module was never imported, so an empty result is still exact.

  Args:
      module_name: The module that defines the classes.
      *type_names: The names of the classes.

  Returns:
      The classes found, or an empty tuple if the module is not imported.
  """
  module = sys.modules.get(module_name)
  return tuple(
      candidate
      for name in type_names
      if isinstance(candidate := getattr(module, name, None), type)
  )


def _is_instance(value: Any, cls: type[Any]) -> bool:
  """Reports whether ``value`` is an instance of ``cls``, running no hooks.

  ``isinstance`` falls back to the value's ``__class__``, which the value can
  override to run code or raise. ``type.__subclasscheck__`` walks the MRO of
  the value's actual type instead, which only ``type`` maintains, so it runs
  no code of the value, its class or its metaclass, and cannot raise.
  """
  return type.__subclasscheck__(cls, type(value))


def _exact_str(value: Any) -> Optional[str]:
  """Returns ``value`` as an exact ``str``, or None if it is not a string.

  A ``str`` subclass can override comparison and truth testing, so it is
  copied into a plain ``str`` with ``str.__str__``, which runs none of the
  subclass's own methods.
  """
  if _is_instance(value, str):
    return str.__str__(value)
  return None


def _instance_dict(obj: Any, cls: type[Any]) -> Optional[dict[str, Any]]:
  """Returns the ``__dict__`` of ``obj``, an instance of ``cls``.

  It's read through the ``__dict__`` descriptor that ``cls`` inherits rather
  than through ``obj``, so a ``__getattribute__`` or a ``__dict__`` property
  defined by a subclass of ``cls`` cannot intercept the read.
  """
  for base in cls.__mro__:
    descriptor = base.__dict__.get("__dict__")
    if descriptor is not None:
      fields = descriptor.__get__(obj, type(obj))
      return fields if type(fields) is dict else None
  return None


def _mcp_model_fields(result: Any) -> Optional[dict[str, Any]]:
  """Returns the fields of an MCP ``CallToolResult`` model, or None."""
  for module_name in _MCP_TYPES_MODULES:
    for model_type in _imported_types(module_name, "CallToolResult"):
      if _is_instance(result, model_type):
        return _instance_dict(result, model_type)
  return None


def _result_fields(result: Any) -> dict[str, Any]:
  """Returns the entries of a tool result that the built-in rules read.

  A dict result, or the fields of an MCP ``CallToolResult`` model, is scanned
  for the entries named in ``_RESULT_FIELD_NAMES`` whose key is an exact
  ``str``. Scanning, instead of looking keys up, never hashes or compares one
  of the result's own keys, so a key that collides with a wanted one and
  raises when compared is simply skipped. Any other result has no entries.

  No method of the result, of its keys or of its values runs here, and no
  entry is interpreted yet, so a malformed entry cannot hide another.
  """
  entries = result if _is_instance(result, dict) else _mcp_model_fields(result)
  if entries is None:
    return {}
  return {
      key: value
      for key, value in dict.items(entries)
      if type(key) is str and key in _RESULT_FIELD_NAMES
  }


def _retry_answer_error_message(fields: dict[str, Any]) -> Optional[str]:
  """Returns the message of a ``ReflectAndRetryToolPlugin`` answer, or None.

  The answer, identified by its ``response_type``, means the tool raised and
  the retry plugin answered the error before the reading plugin's
  ``on_tool_error_callback`` saw it. Its ``error_details`` is ``str(error)``,
  the text ``on_tool_error_callback`` records for the same exception; an
  exception without a message leaves it empty, and the exception's type name
  is recorded instead.
  """
  if _exact_str(fields.get("response_type")) != REFLECT_AND_RETRY_RESPONSE_TYPE:
    return None
  for key in ("error_details", "error_type"):
    message = _exact_str(fields.get(key))
    if message:
      return message
  return _REFLECT_AND_RETRY_TOOL_ERROR_MESSAGE


def _mcp_error_message(fields: dict[str, Any]) -> Optional[str]:
  """Returns the message of a result with an MCP error flag set, or None.

  The flag is the top-level ``isError`` (``is_error`` in MCP SDK 2.x) of any
  dict result, which covers an MCP ``CallToolResult`` dump, or of a
  ``CallToolResult`` model, and it must be exactly ``True``. The result's text
  is tool output, which a plugin's content controls may govern where an error
  message bypasses them, so the message is fixed.
  """
  if fields.get("isError") is True or fields.get("is_error") is True:
    return _MCP_TOOL_ERROR_MESSAGE
  return None


def _apply_rule(
    rule: Callable[[dict[str, Any]], Optional[str]],
    fields: dict[str, Any],
    *,
    log: logging.Logger = logger,
) -> Optional[str]:
  """Applies one built-in rule on its own.

  The rules read separate entries, so one that fails on its entry must not
  keep another from reporting its signal. The warning is constant, and is
  logged after the except block: inside it, a failing log handler would print
  the caught exception, which can carry the result.

  Args:
      rule: The rule to apply.
      fields: The entries ``_result_fields`` read from the result.
      log: The logger of the plugin that applies the rule.

  Returns:
      The rule's error message, or None if the rule found no error or failed.
  """
  try:
    return rule(fields)
  except Exception:  # pylint: disable=broad-except
    pass
  log.warning(
      "Built-in tool result rule %s failed; skipping it.", rule.__name__
  )
  return None


@dataclass(frozen=True)
class ToolResultClassification:
  """A ``tool_result_classifier`` verdict on a tool result.

  For example, to record a tool that reports failure in a ``status`` field:

  ```python
  def classify(*, tool, tool_args, tool_context, result):
    if isinstance(result, dict) and result.get("status") == "ERROR":
      return ToolResultClassification(status="ERROR", error_message="failed")
    return None  # Leave every other result to the built-in rule.

  config = BigQueryLoggerConfig(tool_result_classifier=classify)
  plugin = LoggingPlugin(tool_result_classifier=classify)
  ```

  Attributes:
      status: ``"ERROR"`` records the call as a tool error, for example a
        ``TOOL_ERROR`` row with status ``ERROR`` in
        ``BigQueryAgentAnalyticsPlugin``. ``"OK"`` records it as a completed
        call, even when the built-in MCP rule would have recorded a failure.
      error_message: The error message recorded for ``"ERROR"``, ignored for
        ``"OK"``. A fixed message is recorded when it is empty. A plugin's
        content controls, such as the ``content_formatter`` and
        ``payload_column_denylist`` of ``BigQueryAgentAnalyticsPlugin``, do not
        apply to it, so copy tool output into it only where storing that
        output is acceptable.

  Raises:
      ValueError: If ``status`` is not ``"OK"`` or ``"ERROR"``, or
        ``error_message`` is neither a ``str`` nor None.
  """

  status: Literal["OK", "ERROR"]
  error_message: Optional[str] = None

  def __post_init__(self) -> None:
    if _exact_str(self.status) not in ("OK", "ERROR"):
      raise ValueError("status must be 'OK' or 'ERROR'.")
    if self.error_message is not None and not _is_instance(
        self.error_message, str
    ):
      raise ValueError("error_message must be a str or None.")


class ToolResultClassifier(Protocol):
  """The signature of a plugin's ``tool_result_classifier``.

  The classifier receives, by keyword, the arguments ``after_tool_callback``
  receives, and returns a ``ToolResultClassification``, or None to leave the
  result to the built-in rule. It must be synchronous, with a signature that
  ``inspect`` can read, and must not modify its arguments: the model and the
  agent use them next. It is not called for a ``ReflectAndRetryToolPlugin``
  answer, which is always recorded as the raised error it stands for.
  """

  def __call__(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: Any,
  ) -> Optional[ToolResultClassification]:
    ...


def _classification_error_message(classification: Any) -> Optional[str]:
  """Returns the error message a ``tool_result_classifier`` verdict asks for.

  Only exact ``str`` values are compared or tested, so a verdict whose fields
  override comparison or truth testing cannot run code of its own here. Its
  validation can be bypassed, for example with ``object.__setattr__``, so the
  fields are checked again.

  Args:
      classification: A value the classifier returned, other than None.

  Returns:
      The error message to record, or None for status ``"OK"``.

  Raises:
      TypeError: If ``classification`` is not a ``ToolResultClassification``
        or its ``error_message`` is not a str or None.
      ValueError: If its status is not ``"OK"`` or ``"ERROR"``.
  """
  if not isinstance(classification, ToolResultClassification):
    raise TypeError("tool_result_classifier returned an unsupported value.")
  status = _exact_str(classification.status)
  if status == "OK":
    return None
  if status != "ERROR":
    raise ValueError("ToolResultClassification.status is not OK or ERROR.")
  message = classification.error_message
  if message is None:
    return _CLASSIFIED_TOOL_ERROR_MESSAGE
  text = _exact_str(message)
  if text is None:
    raise TypeError("ToolResultClassification.error_message is not a str.")
  return text or _CLASSIFIED_TOOL_ERROR_MESSAGE


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
  """Marks the exception of a done future as retrieved.

  asyncio logs an exception that nobody retrieved, with its traceback, when
  the future is collected, and the exception of a future a classifier returns
  can carry the tool result.
  """
  if not future.cancelled():
    future.exception()


def _release_awaitable(awaitable: Any) -> None:
  """Releases an awaitable a classifier returned, without awaiting it.

  A coroutine, native or generator-based, is closed, so that Python does not
  warn it was never awaited. An asyncio future or task that is still pending
  is cancelled, so a task that has not started never runs, and the exception
  it ends with is retrieved, as is the exception of one already done. A task
  that ignores the cancellation and never ends is still reported by asyncio
  when it is destroyed. Any other awaitable only runs code when it is awaited,
  so it is left as it is, along with any future it wraps.
  """
  if inspect.iscoroutine(awaitable) or inspect.isgenerator(awaitable):
    awaitable.close()
  elif asyncio.isfuture(awaitable):
    if awaitable.done():
      _retrieve_exception(awaitable)
    else:
      awaitable.add_done_callback(_retrieve_exception)
      awaitable.cancel()


def _signature_without_annotations(function: Any) -> inspect.Signature:
  """Returns the signature of ``function`` without evaluating its annotations.

  From Python 3.14, annotations are evaluated only when they are read, so a
  function may annotate its parameters with names it imports only for type
  checking, and reading those annotations as values raises NameError. Only the
  parameters matter here, so on 3.14 the annotations are read as source
  strings, which evaluates none of them.
  """
  if sys.version_info >= (3, 14):
    import annotationlib  # pylint: disable=g-import-not-at-top

    return inspect.signature(
        function, annotation_format=annotationlib.Format.STRING
    )
  return inspect.signature(function)


def _validate_tool_result_classifier(classifier: Any) -> None:
  """Validates a plugin's ``tool_result_classifier``.

  The classifier runs synchronously inside ``after_tool_callback``, so an
  asynchronous one would only return a coroutine that is never awaited, or an
  async generator that is never iterated. Its signature is checked so that a
  classifier that could never be called as the plugin calls it fails here
  rather than on every tool call. The check binds the keyword arguments to the
  signature, so a callable whose signature takes any arguments, as ``**kwargs``
  does, passes it even if its body then rejects them; such a failure is handled
  at run time like any other.

  Raises:
      ValueError: If the classifier is not callable, is a coroutine or async
        generator function (or its ``__call__`` is, or that of the callable
        a ``functools.partial`` wraps), has no signature that ``inspect`` can
        read, or cannot take the keyword arguments ``tool``, ``tool_args``,
        ``tool_context`` and ``result``.
  """
  if classifier is None:
    return
  if not callable(classifier):
    raise ValueError(
        "tool_result_classifier must be callable, got"
        f" {type(classifier).__name__}."
    )
  # A functools.partial calls the callable it wraps, whose own __call__ may be
  # asynchronous even though the partial's is not.
  wrapped = classifier
  while isinstance(wrapped, functools.partial):
    wrapped = wrapped.func
  if any(
      inspect.iscoroutinefunction(function)
      or inspect.isasyncgenfunction(function)
      for function in (
          classifier,
          getattr(classifier, "__call__", None),
          getattr(wrapped, "__call__", None),
      )
  ):
    raise ValueError(
        "tool_result_classifier must be synchronous; after_tool_callback"
        " calls it without awaiting or iterating the result."
    )
  try:
    signature = _signature_without_annotations(classifier)
  except Exception as e:  # pylint: disable=broad-except
    raise ValueError(
        "tool_result_classifier must have a signature that inspect can read,"
        " so that the plugin can check it takes the keyword arguments tool,"
        " tool_args, tool_context and result. Wrap it in a function that"
        " does."
    ) from e
  try:
    signature.bind(tool=None, tool_args=None, tool_context=None, result=None)
  except TypeError as e:
    raise ValueError(
        "tool_result_classifier must accept the keyword arguments tool,"
        " tool_args, tool_context and result."
    ) from e


def _classify_tool_result_error(
    *,
    tool: BaseTool,
    tool_args: dict[str, Any],
    tool_context: ToolContext,
    result: Any,
    tool_result_classifier: Optional[ToolResultClassifier] = None,
    log: logging.Logger = logger,
) -> Optional[str]:
  """Returns the error message for a tool result that reports a failure.

  Three checks run in turn, each on the entries ``_result_fields`` reads
  without running any of the result's own code:

  1. A ``ReflectAndRetryToolPlugin`` answer is a raised error, recorded as a
     tool error before the classifier runs: with the reading plugin registered
     first, its ``on_tool_error_callback`` records the same error without
     consulting the classifier, so neither plugin order lets a classifier hide
     it.
  2. ``tool_result_classifier``. Its verdict is read inside the ``Exception``
     boundary that surrounds the call, so a classifier that raises or returns
     anything but a readable ``ToolResultClassification`` or None is skipped,
     and the warning names only the tool: the exception, its traceback and the
     returned value can all carry the result. A returned awaitable is ignored
     too, once ``_release_awaitable`` has closed it if it is a coroutine, or
     cancelled it and retrieved its exception if it is an asyncio future or
     task, so that Python does not warn about the coroutine and asyncio does
     not later log an exception the future or task holds. A ``BaseException``
     that is not an ``Exception``, such as a cancellation,
     ``KeyboardInterrupt`` or ``SystemExit``, propagates, because absorbing it
     would keep a task or the process from stopping.
  3. A result whose top-level ``isError`` or ``is_error`` is exactly ``True``:
     any dict, which covers an MCP ``CallToolResult`` dump, or an MCP
     ``CallToolResult`` model.

  Any other result succeeded, including one that merely has an
  ``error_details`` key. The tools' own ``_detect_error_in_response`` hooks,
  which set the OpenTelemetry ``error.type``, are not consulted: several flag
  any dict with a truthy ``error`` key, which a function tool may also return
  as data.

  An exception that escapes, which none of the checks lets through, is left
  to the calling plugin's own boundary.

  Args:
      tool: The tool that produced the result.
      tool_args: The arguments passed to the tool.
      tool_context: The tool context.
      result: The result the model receives.
      tool_result_classifier: The plugin's classifier, if it has one.
      log: The logger of the calling plugin, which the warnings go to.

  Returns:
      The error message to record, or None to record the result as a
      completed call.
  """
  fields = _result_fields(result)
  retry_answer_message = _apply_rule(
      _retry_answer_error_message, fields, log=log
  )
  if retry_answer_message is not None:
    return retry_answer_message
  if tool_result_classifier is not None:
    classifier_failed = False
    try:
      classification = tool_result_classifier(
          tool=tool,
          tool_args=tool_args,
          tool_context=tool_context,
          result=result,
      )
      if inspect.isawaitable(classification):
        _release_awaitable(classification)
        log.warning(
            "tool_result_classifier returned an awaitable for tool %s; it"
            " must be synchronous. Applying the built-in rule instead.",
            tool.name,
        )
      elif classification is not None:
        return _classification_error_message(classification)
    except Exception:  # pylint: disable=broad-except
      classifier_failed = True
    # Logged after the except block: inside it, a failing log handler would
    # print the caught exception, which can carry the result.
    if classifier_failed:
      log.warning(
          "tool_result_classifier failed or returned an unusable verdict for"
          " tool %s; applying the built-in rule instead.",
          tool.name,
      )
  return _apply_rule(_mcp_error_message, fields, log=log)


# For each plugin instance, keyed by id(), the tool call whose failure its
# on_tool_error_callback recorded last. When a plugin registered after it
# answers that error (ReflectAndRetryToolPlugin, for example), the answer
# reaches after_tool_callback as the call's result, and that callback must not
# record the failure again. The key keeps two plugins in one runner from
# consuming each other's entry: one registered after the answering plugin
# never saw the error and must still record the answer. The mapping is replaced
# rather than mutated, so a sibling task holding a copy of this context never
# sees the change. The references are weak, so a call that ends without
# after_tool_callback does not keep its context alive.
_tool_errors_recorded_ctx: contextvars.ContextVar[
    collections.abc.Mapping[int, weakref.ref[object]]
] = contextvars.ContextVar(
    "_adk_tool_errors_recorded", default=MappingProxyType({})
)


def _mark_tool_error_recorded(plugin: object, tool_context: object) -> None:
  """Marks that ``plugin`` recorded the failure of this tool call.

  Called from ``on_tool_error_callback`` once nothing that can fail remains
  between the mark and the record, because ``after_tool_callback`` skips a
  marked call.

  Args:
      plugin: The plugin that recorded the failure.
      tool_context: The context of the failed call.
  """
  try:
    recorded_call = weakref.ref(tool_context)
  except TypeError:
    # No ADK flow passes a context that cannot take a weak reference. Such a
    # call goes unmarked: a later plugin's answer to this error may then be
    # recorded again, but the error's own record is not lost.
    return
  _tool_errors_recorded_ctx.set({
      **_tool_errors_recorded_ctx.get(),
      id(plugin): recorded_call,
  })


def _pop_tool_error_recorded(plugin: object, tool_context: object) -> bool:
  """Reports, and forgets, whether ``plugin`` recorded this call's failure.

  ``after_tool_callback`` calls it first: when it returns True, the result is
  the answer a later plugin gave to an error that ``on_tool_error_callback``
  already recorded, and the call must not be recorded again.

  Args:
      plugin: The plugin whose ``after_tool_callback`` runs.
      tool_context: The context of the call.

  Returns:
      Whether ``plugin`` marked this call with ``_mark_tool_error_recorded``.
  """
  recorded = _tool_errors_recorded_ctx.get()
  recorded_call = recorded.get(id(plugin))
  if recorded_call is None or recorded_call() is not tool_context:
    return False
  _tool_errors_recorded_ctx.set(
      {key: call for key, call in recorded.items() if key != id(plugin)}
  )
  return True
