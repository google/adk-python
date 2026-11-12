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
import collections.abc
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Mapping
import contextvars
import inspect
import logging
import threading
import typing
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic import PydanticSchemaGenerationError
from pydantic import PydanticUserError
from pydantic import TypeAdapter
from typing_extensions import override

from ..auth.auth_tool import AuthConfig
from ..events.event import Event
from ..events.request_input import RequestInput
from ..platform.thread import create_thread
from ..utils._callable_utils import CallableSpec
from ..utils._schema_utils import _unwrap_content_payload
from ..utils._schema_utils import annotation_expects_str
from ..utils._sync_runner import _invoke_callable
from ..utils._sync_runner import _resolve_sync_callable_runner
from ..utils._sync_runner import _use_sync_callable_runner
from ..utils.context_utils import Aclosing
from ._base_node import BaseNode
from ._errors import WorkflowConfigurationError
from ._errors import WorkflowDataError
from ._retry_config import RetryConfig
from .utils._workflow_hitl_utils import apply_confirmation_resume
from .utils._workflow_hitl_utils import auth_interrupt_id
from .utils._workflow_hitl_utils import create_auth_request_event
from .utils._workflow_hitl_utils import create_confirmation_request_input
from .utils._workflow_hitl_utils import has_auth_credential
from .utils._workflow_hitl_utils import process_auth_resume

logger = logging.getLogger("google_adk." + __name__)
_ANY_TYPE_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


async def _sync_to_async_gen(
    sync_gen: collections.abc.Generator[Any, None, None],
) -> AsyncGenerator[Any, None]:
  """Wraps a synchronous generator as an async generator.

  When a sync callable runner (the tool thread pool) is bound, every
  ``next()`` and the final ``close()`` run on one dedicated thread, so
  thread-local state survives across yields. Each step occupies a pool worker
  until it finishes, so ``max_workers`` still bounds how many generator steps
  execute at once; between yields the dedicated thread is parked and holds no
  pool worker, so suspended generators cannot starve the pool.
  """
  runner = _resolve_sync_callable_runner()
  if runner is None:
    try:
      for item in sync_gen:
        yield item
    finally:
      sync_gen.close()
    return

  sentinel = object()
  gen_context = contextvars.copy_context()
  # Guards gen_thread, closed and finished, which the event loop, the pool
  # worker running a step and the generator thread all read.
  lock = threading.Lock()
  want_next = threading.Event()
  step_done = threading.Event()
  closed = False
  finished = False
  close_error: BaseException | None = None
  step_result: tuple[str, Any] = ("done", None)
  gen_thread: Thread | None = None

  def _drive_sync_gen() -> None:
    nonlocal step_result, finished, close_error
    # Like any sync callable on the pool, the generator body runs with no
    # runner bound, so a nested sync call runs inline on its thread instead of
    # waiting for the pool worker that this step is holding.
    with _use_sync_callable_runner(None):
      try:
        while True:
          want_next.wait()
          want_next.clear()
          with lock:
            if closed:
              break
          try:
            item = next(sync_gen, sentinel)
          except BaseException as exc:  # pylint: disable=broad-exception-caught
            step_result = ("error", exc)
            break
          if item is sentinel:
            step_result = ("done", None)
            break
          step_result = ("item", item)
          step_done.set()
      finally:
        try:
          sync_gen.close()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
          close_error = exc
        finally:
          with lock:
            finished = True
          step_done.set()

  def _step_in_pool() -> tuple[str, Any]:
    # Runs on a pool worker and blocks it until the generator thread finishes
    # this step, which is what keeps generator execution within max_workers.
    nonlocal gen_thread
    with lock:
      if closed or finished:
        return ("done", None)
      if gen_thread is None:
        gen_thread = create_thread(gen_context.run, _drive_sync_gen)
        gen_thread.name = (
            f"adk_sync_generator_for_{threading.current_thread().name}"
        )
        gen_thread.daemon = True
        gen_thread.start()
      # Cleared under the lock, before the generator thread can mark itself
      # finished, so its final step_done.set() is never lost.
      step_done.clear()
      want_next.set()
    step_done.wait()
    with lock:
      is_finished = finished
      is_closed = closed
    if is_finished:
      gen_thread.join()
    return ("done", None) if is_closed else step_result

  waiting_for_step = False
  step_task: asyncio.Future[tuple[str, Any]] | None = None
  try:
    while True:
      step_task = asyncio.ensure_future(runner(_step_in_pool, {}))
      waiting_for_step = True
      kind, payload = await step_task
      waiting_for_step = False
      if kind == "done":
        break
      if kind == "error":
        raise payload
      yield payload
  finally:
    # Once closed is set, no step starts the generator thread, so whichever
    # side observes gen_thread is None owns closing the generator.
    with lock:
      closed = True
    want_next.set()
    if waiting_for_step and step_task is not None:
      # A step may still be inside next(); the generator thread closes the
      # generator once next() returns, without blocking this cancellation.
      step_task.cancel()
      try:
        await step_task
      except asyncio.CancelledError:
        pass
    with lock:
      thread = gen_thread
    if thread is None:
      sync_gen.close()
    elif not waiting_for_step:
      if thread.is_alive():
        await asyncio.shield(runner(thread.join, {}))
      if close_error is not None:
        raise close_error


if TYPE_CHECKING:
  from threading import Thread

  from ..agents.context import Context

# Output types that are framework control-flow items, not data schemas.
_PASSTHROUGH_OUTPUT_TYPES = (types.Content, Event, RequestInput)

# Generator origins used for unwrapping yield types.
_GENERATOR_ORIGINS = (
    collections.abc.Generator,
    collections.abc.AsyncGenerator,
)


class _ResolvedParameters(dict[str, Any]):
  """Marker dict subclass indicating parameters have already been resolved.

  ``source`` keeps the input they were resolved from. The first bind uses the
  resolved values as they are; a later bind, such as a retry of the same node,
  resolves again from ``source`` so values that coercion rebuilds are fresh
  rather than carried over from a previous attempt.
  """

  source: Any
  bound: bool

  def __init__(self, values: dict[str, Any], *, source: Any = None):
    super().__init__(values)
    self.source = source
    self.bound = False


def _warn_on_non_text_parts(
    content: types.Content, func_name: str, param_name: str
) -> None:
  """Logs a warning when Content contains non-text parts."""
  for part in content.parts or []:
    if part.text is None and (
        part.inline_data or part.file_data or part.executable_code
    ):
      logger.warning(
          'Parameter "%s" of function "%s" expects str but received'
          " Content with non-text parts (e.g. inline_data, file_data)."
          " Non-text parts are dropped during auto-conversion.",
          param_name,
          func_name,
      )


class FunctionNode(BaseNode):
  """A node that wraps a Python sync/async function or generator.

  Type coercions applied to function parameters (via ``TypeAdapter``):
    - ``dict`` → ``BaseModel`` when the annotation is a Pydantic model.
    - ``list[dict]`` → ``list[BaseModel]``, ``dict[K, dict]`` →
      ``dict[K, BaseModel]``, etc.
    - ``types.Content`` → ``str`` when the annotation expects ``str``
      (including ``Optional[str]`` / ``Union[str, ...]``).
    - All other values are validated/coerced by Pydantic's ``TypeAdapter``.
  """

  auth_config: AuthConfig | None = None
  """If set, the framework requests user authentication before running.

  When the node runs for the first time and no credential is found in
  session state, it yields an ``adk_request_credential`` event and
  interrupts.  On resume, the credential is stored and the node
  re-runs with the credential available via
  ``AuthHandler(auth_config).get_auth_response(ctx.state)``.
  """

  require_confirmation: bool | Callable[..., bool] = False
  """Whether this node requires user confirmation before running.

  Can be a boolean or a callable predicate that receives the function's
  bound arguments (and ``ctx`` if declared) and returns a boolean.
  Requires ``rerun_on_resume=True`` so the node reruns after confirmation is
  provided (``@node`` and ``build_node`` enable this automatically when
  ``rerun_on_resume`` is not explicitly set).
  """

  parameter_binding: Literal["state", "node_input"] = "state"
  """How function parameters are bound.

  ``'state'`` (default) binds parameters from ``ctx.state``.
  ``'node_input'`` binds parameters from ``node_input`` dict and infers
  ``input_schema`` / ``output_schema`` from the function signature
  (used when the node acts as an agent's tool).
  """

  # Private attributes (won't be serialized)
  _func: Callable[..., Any] = PrivateAttr()
  _sig: inspect.Signature = PrivateAttr()
  _type_hints: dict[str, Any] = PrivateAttr()
  _type_adapters: dict[str, TypeAdapter[Any]] = PrivateAttr()
  _failed_type_adapters: set[str] = PrivateAttr(default_factory=set)
  # Always assigned in __init__; 'ctx' is the fallback used when the wrapped
  # function declares no context parameter.
  _context_param_name: str = PrivateAttr(default="ctx")

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      name: str | None = None,
      rerun_on_resume: bool = False,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
      auth_config: AuthConfig | None = None,
      require_confirmation: bool | Callable[..., bool] = False,
      parameter_binding: Literal["state", "node_input"] = "state",
      state_schema: type[BaseModel] | None = None,
  ):
    """Initializes FunctionNode.

    Args:
      func: A sync/async function or sync/async generator function that forms
        the node's logic. It can accept 'ctx: Context' and 'node_input: Any' as
        arguments, depending on its signature. If the function is not a
        generator, its return value will be wrapped in an Event, unless the
        return value is None.
      name: The name of the node. If None, it defaults to func.__name__.
      rerun_on_resume: If True, the node will be rerun after being interrupted
        and resumed. If False, the node will be marked as completed and the
        resuming input will be treated as the node's output. Defaults to False.
        Nodes with require_confirmation or auth_config require
        rerun_on_resume=True.
      retry_config: If provided, the node will be retried on failure based on
        this configuration.
      timeout: Maximum time in seconds for this node to complete.
      auth_config: If provided, the framework requests user authentication
        before running the node. Requires rerun_on_resume=True (the node must
        rerun after credentials are provided).
      require_confirmation: Whether this node requires confirmation before
        running. Can be a boolean or a callable predicate returning a boolean.
        Requires rerun_on_resume=True (the node must rerun after confirmation is
        provided).
      parameter_binding: How function parameters are bound. ``'state'``
        (default) binds parameters from ``ctx.state``. ``'node_input'`` binds
        parameters from ``node_input`` dict and infers ``input_schema`` /
        ``output_schema`` from the function signature (used when the node acts
        as an agent's tool).
    """

    if not callable(func):
      raise WorkflowConfigurationError("Function must be callable.")

    if require_confirmation and not rerun_on_resume:
      raise WorkflowConfigurationError(
          "FunctionNode with require_confirmation requires"
          " rerun_on_resume=True. The node must rerun after confirmation is"
          " provided."
      )

    if auth_config and not rerun_on_resume:
      raise WorkflowConfigurationError(
          "FunctionNode with auth_config requires rerun_on_resume=True."
          " The node must rerun after credentials are provided."
      )

    spec = CallableSpec(func)
    inferred_name = (
        name
        or getattr(func, "__name__", None)
        or getattr(spec.unwrapped_func, "__name__", None)
    )
    if not inferred_name:
      raise WorkflowConfigurationError(
          "FunctionNode must have a name. If the wrapped callable does not"
          " have a '__name__' attribute, please provide a name explicitly."
      )

    super().__init__(
        name=inferred_name,
        description=spec.doc,
        rerun_on_resume=rerun_on_resume,
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
        require_confirmation=require_confirmation,
        parameter_binding=parameter_binding,
        state_schema=state_schema,
    )

    sig = spec.signature
    type_hints = spec.get_best_effort_type_hints()

    self._context_param_name = spec.context_param_name or "ctx"

    # Set private attributes
    self._func = func
    self._unwrapped_func = spec.unwrapped_func
    self._sig = sig
    self._type_hints = type_hints
    self._type_adapters = {}
    self._failed_type_adapters = set()
    for name, hint in type_hints.items():
      if name == "return" or name == self._context_param_name:
        continue
      adapter = self._build_type_adapter(name, hint)
      if adapter is not None:
        self._type_adapters[name] = adapter
      else:
        self._failed_type_adapters.add(name)

    # Infer schemas based on the parameter binding mode.
    if parameter_binding == "node_input":
      self._infer_schemas_from_func_signature(func)
    else:
      self._infer_schemas_for_state_mode(type_hints)

  def _infer_schemas_for_state_mode(self, type_hints: dict[str, Any]) -> None:
    """Infers schemas from type hints in state binding mode.

    ``output_schema`` is inferred from the return type hint (unwrapping
    generator types). ``input_schema`` is inferred from the ``node_input``
    parameter type hint.
    """
    # Infer output_schema from the return type hint.
    # For generators (Generator[T, ...] / AsyncGenerator[T, ...]),
    # extract the yield type T as the schema.
    return_hint = type_hints.get("return")
    schema_hint = return_hint

    # Unwrap Generator[T, ...] / AsyncGenerator[T, ...] to T.
    if return_hint is not None:
      origin = typing.get_origin(return_hint)
      if origin in _GENERATOR_ORIGINS:
        args = typing.get_args(return_hint)
        schema_hint = args[0] if args else None

    if (
        schema_hint is not None
        and inspect.isclass(schema_hint)
        and issubclass(schema_hint, BaseModel)
        and not issubclass(schema_hint, _PASSTHROUGH_OUTPUT_TYPES)
    ):
      self.output_schema = schema_hint

    # Infer input_schema from node_input type hint.
    input_hint = type_hints.get("node_input")
    if (
        input_hint is not None
        and inspect.isclass(input_hint)
        and issubclass(input_hint, BaseModel)
    ):
      self.input_schema = input_hint

  def _infer_schemas_from_func_signature(
      self, func: Callable[..., Any]
  ) -> None:
    """Infers input/output schema from the function signature.

    Used when ``parameter_binding='node_input'``. ``input_schema`` is
    built from function parameters (excluding the context parameter),
    ``output_schema`` from the return type hint.
    """
    from ..tools._function_tool_declarations import _build_parameters_json_schema
    from ..tools._function_tool_declarations import _build_response_json_schema

    ignore_params: list[str] = [self._context_param_name]
    try:
      self.input_schema = _build_parameters_json_schema(
          func, ignore_params=ignore_params
      )
    except (
        TypeError,
        NameError,
        PydanticSchemaGenerationError,
        PydanticUserError,
    ) as e:
      logger.warning(
          "Skipping input schema inference for FunctionNode '%s' due to"
          " unhandled annotation type: %s",
          self.name,
          e,
      )
      self.input_schema = None
    response_schema = _build_response_json_schema(func)
    if response_schema is not None:
      self.output_schema = response_schema

  def _validates_by_resolving(self) -> bool:
    """Returns whether input validation delegates to _resolve_parameters."""
    return self.parameter_binding == "node_input" and (
        self.input_schema is None
        or isinstance(self.input_schema, (dict, types.Schema))
    )

  def _resolve_parameters(
      self,
      node_input: Any,
      *,
      state: Any = None,
  ) -> _ResolvedParameters:
    """Resolves and coerces non-context function parameters."""
    input_bound = self.parameter_binding == "node_input"
    if input_bound:
      source: Any = (
          node_input if isinstance(node_input, (dict, BaseModel)) else {}
      )
      source_name = "node_input"
    else:
      source = state if state is not None else {}
      source_name = "state"
    kwargs: dict[str, Any] = {}
    for param_name, param in self._sig.parameters.items():
      if param_name == self._context_param_name:
        continue
      if param.kind == inspect.Parameter.VAR_POSITIONAL:
        continue
      if param.kind == inspect.Parameter.VAR_KEYWORD:
        if input_bound and isinstance(source, dict):
          for k, v in source.items():
            if k not in self._sig.parameters and k != self._context_param_name:
              kwargs[k] = v
        continue

      # In state mode, 'node_input' param is passed through directly.
      if not input_bound and param_name == "node_input":
        value = node_input
        if param_name in self._type_hints:
          value = self._coerce_param(
              param_name,
              node_input,
              self._type_hints[param_name],
          )
        kwargs[param_name] = value
        continue

      has_param = False
      value = None
      if isinstance(source, BaseModel):
        if hasattr(source, param_name):
          has_param = True
          value = getattr(source, param_name)
      else:
        try:
          if param_name in source:
            has_param = True
            value = source[param_name]
        except (TypeError, KeyError):
          pass

      if (
          not has_param
          and input_bound
          and param_name == "node_input"
          and self.input_schema is not None
          and not isinstance(self.input_schema, (dict, types.Schema))
          and param_name in self._type_hints
      ):
        try:
          kwargs[param_name] = self._coerce_param(
              param_name,
              node_input,
              self._type_hints[param_name],
          )
          continue
        except Exception:
          pass

      if has_param:
        if param_name in self._type_hints:
          value = self._coerce_param(
              param_name,
              value,
              self._type_hints[param_name],
          )
        kwargs[param_name] = value
      elif param.default is not inspect.Parameter.empty:
        kwargs[param_name] = param.default
      else:
        raise WorkflowDataError(
            f'Missing value for parameter "{param_name}" of function'
            f' "{self.name}". It was not found in {source_name} and has no'
            " default value."
        )
    return _ResolvedParameters(kwargs, source=node_input)

  def _bind_parameters(self, ctx: Context, node_input: Any) -> dict[str, Any]:
    """Binds function parameters from the appropriate data source.

    In ``'node_input'`` mode, non-context parameters are looked up in the
    ``node_input`` dict.  In ``'state'`` mode, the ``node_input`` parameter
    is passed through directly and all other non-context parameters are
    looked up in ``ctx.state``.
    """
    if (
        self._validates_by_resolving()
        and isinstance(node_input, _ResolvedParameters)
        and not node_input.bound
    ):
      node_input.bound = True
      kwargs: dict[str, Any] = dict(node_input)
    else:
      if isinstance(node_input, _ResolvedParameters):
        # A retry gets the same resolved input the failed attempt may have
        # mutated, so resolve again from the original input.
        node_input = node_input.source
      kwargs = dict(self._resolve_parameters(node_input, state=ctx.state))
    if self._context_param_name in self._sig.parameters:
      kwargs[self._context_param_name] = ctx
    return kwargs

  def _to_event(
      self, ctx: Context, data: object
  ) -> Event | RequestInput | None:
    """Converts a function return value to an Event.

    Pass-through types (returned as-is): Event, RequestInput.
    None is returned as None (caller skips it) unless there are pending
    state changes.
    All other values are wrapped in an Event(output=...).

    State changes made via ``ctx.state`` during function execution are
    captured in ``ctx.actions.state_delta`` and attached to the emitted
    event so that they are persisted by the session service.
    """
    state_delta = (
        dict(ctx.actions.state_delta) if ctx.actions.state_delta else None
    )

    if data is None:
      if state_delta:
        return Event(state=state_delta)
      return None

    if isinstance(data, Event):
      if state_delta:
        data.actions.state_delta.update(state_delta)
      return data
    if isinstance(data, RequestInput):
      return data
    if isinstance(data, types.Content):
      return Event(
          content=data,
          state=state_delta,
      )

    if isinstance(data, BaseModel):
      data = data.model_dump()

    return Event(
        output=data,
        state=state_delta,
    )

  def _build_type_adapter(
      self, param_name: str, annotated_type: Any
  ) -> TypeAdapter[Any] | None:
    """Builds a TypeAdapter for `annotated_type`, returning None when unsupported."""
    try:
      adapter = TypeAdapter(annotated_type)
      if not adapter.pydantic_complete:
        adapter.rebuild()
      return adapter
    except (
        TypeError,
        NameError,
        PydanticSchemaGenerationError,
        PydanticUserError,
    ) as e:
      logger.warning(
          "Skipping coercion for parameter '%s' of FunctionNode '%s' due to"
          " unhandled annotation type '%s': %s",
          param_name,
          self.name,
          annotated_type,
          e,
      )
      return None

  def _coerce_param(
      self,
      param_name: str,
      value: Any,
      annotated_type: Any,
  ) -> Any:
    """Coerces a parameter value to match its type annotation.

    Uses Pydantic's ``TypeAdapter`` for validation and coercion (handles
    ``dict`` → ``BaseModel``, ``list[dict]`` → ``list[BaseModel]``, unions,
    primitives, etc.).  A special case converts ``types.Content`` → ``str``
    when the annotation expects ``str``, or parses JSON from ``types.Content``
    when the annotation expects structured data.

    Args:
      param_name: The name of the parameter (for error messages).
      value: The value to coerce.
      annotated_type: The type annotation of the parameter.

    Returns:
      The coerced value.
    """
    # Content → str / JSON auto-conversion (e.g. user content from START node).
    if isinstance(value, types.Content):
      if annotation_expects_str(annotated_type):
        _warn_on_non_text_parts(value, self.name, param_name)
      value = _unwrap_content_payload(value, annotated_type)
    if param_name in self._failed_type_adapters:
      return value
    adapter = self._type_adapters.get(param_name)
    if adapter is None:
      adapter = self._build_type_adapter(param_name, annotated_type)
      if adapter is None:
        self._failed_type_adapters.add(param_name)
        return value
      self._type_adapters[param_name] = adapter
    return adapter.validate_python(value)

  @override
  def _validate_input_data(self, data: Any) -> Any:
    """Validates input data for FunctionNode."""
    if self._validates_by_resolving():
      if isinstance(data, _ResolvedParameters):
        return data
      return self._resolve_parameters(data)

    return super()._validate_input_data(data)

  @override
  def _validate_output_data(self, data: Any) -> Any:
    """Validates output data for FunctionNode, bypassing schema check on rejection."""
    if self.require_confirmation and data == {
        "error": "This tool call is rejected."
    }:
      return data
    return super()._validate_output_data(data)

  @override
  def model_copy(
      self, *, update: Mapping[str, Any] | None = None, deep: bool = False
  ) -> FunctionNode:
    copied = super().model_copy(update=update, deep=deep)
    if not update or "name" not in update:
      return copied

    # If the wrapped function is a bound method of a Node, we need to clone
    # the Node and re-bind the function to the new instance.
    # This is needed if the function is referring to params like 'name' from the "self" reference.
    # Like Workflow or LLM use that name for event node_paths or retrieving session events.
    func = self._func
    if inspect.ismethod(func) and isinstance(
        getattr(func, "__self__", None), BaseNode
    ):
      method_self = getattr(func, "__self__")
      method_name = getattr(func, "__name__")

      # Pass the name update to the cloned agent instance if it's being passed
      # to the FunctionNode (case for parallel workers).
      agent_update = {
          "name": update["name"],
      }

      new_obj = method_self.model_copy(update=agent_update)
      copied._func = getattr(new_obj, method_name)
    else:
      copied._func = func

    copied._sig = self._sig
    copied._unwrapped_func = self._unwrapped_func
    copied._type_hints = self._type_hints
    copied._type_adapters = self._type_adapters
    copied._context_param_name = self._context_param_name
    return copied

  async def _check_require_confirmation(
      self,
      ctx: Context,
      *,
      node_input: Any = None,
      bound_args: dict[str, Any],
  ) -> bool:
    """Returns whether this node requires confirmation for the given arguments."""
    if not callable(self.require_confirmation):
      return bool(self.require_confirmation)

    pred_spec = CallableSpec(self.require_confirmation)
    args_to_call = dict(bound_args)
    if pred_spec.has_signature:
      valid_params = set(pred_spec.signature.parameters.keys())
      has_kwargs = any(
          p.kind == inspect.Parameter.VAR_KEYWORD
          for p in pred_spec.signature.parameters.values()
      )
      context_param = pred_spec.context_param_name or (
          "ctx" if "ctx" in valid_params else "tool_context"
      )
      if context_param in valid_params:
        args_to_call[context_param] = ctx
      if "node_input" in valid_params and "node_input" not in args_to_call:
        args_to_call["node_input"] = node_input
      for param_name in valid_params:
        if (
            param_name not in args_to_call
            and param_name != context_param
            and param_name in ctx.state
        ):
          args_to_call[param_name] = ctx.state[param_name]
      if not has_kwargs:
        args_to_call = {
            k: v for k, v in args_to_call.items() if k in valid_params
        }
    return bool(
        await _invoke_callable(
            self.require_confirmation,
            args_to_call,
            unwrapped=pred_spec.unwrapped_func,
        )
    )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    kwargs = self._bind_parameters(ctx, node_input)

    # --- Confirmation gate ---
    resumed_confirmation = False
    if self.require_confirmation:
      resumed_confirmation = apply_confirmation_resume(ctx)
      if ctx.tool_confirmation is not None:
        if not ctx.tool_confirmation.confirmed:
          yield Event(output={"error": "This tool call is rejected."})
          return
      else:
        bound_args = {
            k: v for k, v in kwargs.items() if k != self._context_param_name
        }
        if await self._check_require_confirmation(
            ctx, node_input=node_input, bound_args=bound_args
        ):
          yield create_confirmation_request_input(
              node_path=ctx.node_path,
              tool_name=self.name,
              args=_ANY_TYPE_ADAPTER.dump_python(bound_args),
          )
          return

    # --- Auth gate ---
    if self.auth_config:
      interrupt_id = auth_interrupt_id(ctx.node_path)
      auth_response = ctx.resume_inputs.get(interrupt_id)
      if auth_response is not None:
        await process_auth_resume(
            auth_response, self.auth_config, ctx.state, interrupt_id
        )
      elif not has_auth_credential(self.auth_config, ctx.state):
        yield create_auth_request_event(
            self.auth_config, interrupt_id, ctx.state
        )
        return

    unwrapped_func = self._unwrapped_func
    if inspect.isasyncgenfunction(unwrapped_func):
      items = self._func(**kwargs)
    elif inspect.isgeneratorfunction(unwrapped_func):
      items = _sync_to_async_gen(self._func(**kwargs))
    else:
      items = None

    emitted_event = False
    if items is not None:
      async with Aclosing(items) as items:
        async for item in items:
          event = self._to_event(ctx, item)
          if event is not None:
            emitted_event = True
            yield event
    else:
      result = await _invoke_callable(
          self._func, kwargs, unwrapped=unwrapped_func
      )
      event = self._to_event(ctx, result)
      if event is not None:
        emitted_event = True
        yield event

    if not emitted_event and resumed_confirmation:
      yield Event()

  def _as_tool_node(self) -> FunctionNode:
    """Returns a FunctionNode clone adapted for tool execution (node_input binding)."""
    if self.parameter_binding == "node_input":
      return self
    func = getattr(self, "_func", None)
    if not callable(func):
      raise ValueError(
          f"FunctionNode '{self.name}' has no underlying callable to adapt as a"
          " tool node."
      )
    node = FunctionNode(
        func=func,
        name=self.name,
        rerun_on_resume=self.rerun_on_resume,
        retry_config=self.retry_config,
        timeout=self.timeout,
        auth_config=self.auth_config,
        require_confirmation=self.require_confirmation,
        parameter_binding="node_input",
        state_schema=self.state_schema,
    )
    node.description = self.description
    node.wait_for_output = self.wait_for_output
    if self.input_schema is not None:
      node.input_schema = self.input_schema
    if self.output_schema is not None:
      node.output_schema = self.output_schema
    return node
