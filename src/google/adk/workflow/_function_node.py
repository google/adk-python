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

import collections.abc
from collections.abc import AsyncGenerator
from collections.abc import Callable
import functools
import inspect
import logging
import typing
from typing import Any
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic import PydanticSchemaGenerationError
from pydantic import TypeAdapter
from typing_extensions import override

from ..auth.auth_tool import AuthConfig
from ..events.event import Event
from ..events.request_input import RequestInput
from ._base_node import BaseNode
from ._retry_config import RetryConfig
from .utils._workflow_hitl_utils import create_auth_request_event
from .utils._workflow_hitl_utils import has_auth_credential
from .utils._workflow_hitl_utils import process_auth_resume

logger = logging.getLogger('google_adk.' + __name__)


async def _sync_to_async_gen(
    sync_gen: collections.abc.Generator[Any, None, None],
) -> AsyncGenerator[Any, None]:
  """Wraps a synchronous generator as an async generator."""
  for item in sync_gen:
    yield item


if TYPE_CHECKING:
  from ..agents.context import Context

# Output types that are framework control-flow items, not data schemas.
_PASSTHROUGH_OUTPUT_TYPES = (types.Content, Event, RequestInput)

# Generator origins used for unwrapping yield types.
_GENERATOR_ORIGINS = (
    collections.abc.Generator,
    collections.abc.AsyncGenerator,
)


def _unwrap_callable(func: Callable[..., Any]) -> Callable[..., Any]:
  """Unwraps partials, bound methods and callable objects to find the stable underlying function."""
  while True:
    if isinstance(func, functools.partial):
      func = func.func
    elif hasattr(func, '__func__'):  # bound method
      func = func.__func__
    elif (
        hasattr(func, '__call__')
        and not inspect.isfunction(func)
        and not inspect.ismethod(func)
    ):
      # callable object, unwrap to its __call__ method
      func = func.__call__
    else:
      break
  return func


@functools.lru_cache(maxsize=1024)
def _get_type_hints_for_unwrapped(func: Callable[..., Any]) -> dict[str, Any]:
  """Cached version of typing.get_type_hints."""
  try:
    return typing.get_type_hints(func)
  except (TypeError, NameError, AttributeError):
    return {}


def _get_type_hints_cached(func: Callable[..., Any]) -> dict[str, Any]:
  """Cached version of typing.get_type_hints with robust unwrapping."""
  unwrapped = _unwrap_callable(func)
  return _get_type_hints_for_unwrapped(unwrapped)


def _content_to_str(
    content: types.Content, func_name: str, param_name: str
) -> str:
  """Extracts text from a Content object, warning on non-text parts."""
  texts = []
  for part in content.parts or []:
    if part.text is not None:
      texts.append(part.text)
    elif part.inline_data or part.file_data or part.executable_code:
      logger.warning(
          'Parameter "%s" of function "%s" expects str but received'
          ' Content with non-text parts (e.g. inline_data, file_data).'
          ' Non-text parts are dropped during auto-conversion.',
          param_name,
          func_name,
      )
  return ''.join(texts)


def _expects_str(annotated_type: Any) -> bool:
  """Returns True if the annotation is or contains ``str``."""
  if annotated_type is str:
    return True
  if typing.get_origin(annotated_type) is typing.Union:
    return any(_expects_str(a) for a in typing.get_args(annotated_type))
  return False


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

  parameter_binding: Literal['state', 'node_input'] = 'state'
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
  _type_adapters: dict[str, TypeAdapter] = PrivateAttr()
  _context_param_name: str | None = PrivateAttr(default=None)

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      name: str | None = None,
      rerun_on_resume: bool = False,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
      auth_config: AuthConfig | None = None,
      parameter_binding: Literal['state', 'node_input'] = 'state',
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
        resuming input will be treated as the node's output.
      retry_config: If provided, the node will be retried on failure based on
        this configuration.
      timeout: Maximum time in seconds for this node to complete.
      auth_config: If provided, the framework requests user authentication
        before running the node. Requires rerun_on_resume=True (the node
        must rerun after credentials are provided).
      parameter_binding: How function parameters are bound. ``'state'``
        (default) binds parameters from ``ctx.state``. ``'node_input'``
        binds parameters from ``node_input`` dict and infers
        ``input_schema`` / ``output_schema`` from the function signature
        (used when the node acts as an agent's tool).
    """

    if not callable(func):
      raise TypeError('Function must be callable.')

    if auth_config and not rerun_on_resume:
      raise ValueError(
          'FunctionNode with auth_config requires rerun_on_resume=True.'
          ' The node must rerun after credentials are provided.'
      )

    inferred_name = (
        name
        or getattr(func, '__name__', None)
        or getattr(_unwrap_callable(func), '__name__', None)
    )
    if not inferred_name:
      raise ValueError(
          'FunctionNode must have a name. If the wrapped callable does not'
          " have a '__name__' attribute, please provide a name explicitly."
      )

    super().__init__(
        name=inferred_name,
        description=inspect.getdoc(func) or '',
        rerun_on_resume=rerun_on_resume,
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
        parameter_binding=parameter_binding,
        state_schema=state_schema,
    )

    sig = inspect.signature(func)
    type_hints = _get_type_hints_cached(func)

    # Detect the context parameter name (e.g. 'ctx', 'tool_context').
    from ..utils.context_utils import find_context_parameter

    self._context_param_name = find_context_parameter(func) or 'ctx'

    # Set private attributes
    self._func = func
    self._sig = sig
    self._type_hints = type_hints
    self._type_adapters = {}
    for name, hint in type_hints.items():
      if name == 'return' or name == self._context_param_name:
        continue
      try:
        self._type_adapters[name] = TypeAdapter(hint)
      except (TypeError, PydanticSchemaGenerationError):
        pass

    # Infer schemas based on the parameter binding mode.
    if parameter_binding == 'node_input':
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
    return_hint = type_hints.get('return')
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
    input_hint = type_hints.get('node_input')
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

    ignore_params: list[str] = (
        [self._context_param_name] if self._context_param_name else []
    )
    self.input_schema = _build_parameters_json_schema(
        func, ignore_params=ignore_params
    )
    response_schema = _build_response_json_schema(func)
    if response_schema is not None:
      self.output_schema = response_schema

  def _bind_parameters(self, ctx: Context, node_input: Any) -> dict[str, Any]:
    """Binds function parameters from the appropriate data source.

    In ``'node_input'`` mode, non-context parameters are looked up in the
    ``node_input`` dict.  In ``'state'`` mode, the ``node_input`` parameter
    is passed through directly and all other non-context parameters are
    looked up in ``ctx.state``.
    """
    input_bound = self.parameter_binding == 'node_input'
    if input_bound:
      source = node_input if isinstance(node_input, dict) else {}
    else:
      source = ctx.state
    source_name = 'node_input' if input_bound else 'state'

    kwargs: dict[str, Any] = {}
    for param_name, param in self._sig.parameters.items():
      if param_name == self._context_param_name:
        kwargs[param_name] = ctx
        continue

      # In state mode, 'node_input' param is passed through directly.
      if not input_bound and param_name == 'node_input':
        value = node_input
        if param_name in self._type_hints:
          value = self._coerce_param(
              param_name,
              node_input,
              self._type_hints[param_name],
          )
        kwargs[param_name] = value
        continue

      if param_name in source:
        value = source[param_name]
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
        raise ValueError(
            f'Missing value for parameter "{param_name}" of function'
            f' "{self.name}". It was not found in {source_name} and has no'
            ' default value.'
        )
    return kwargs

  def _to_event(self, ctx: Context, data: Any) -> Event | None:
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
      if data.output is not None:
        data.output = self._validate_output_data(data.output)
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

    data = self._validate_output_data(data)

    return Event(
        output=data,
        state=state_delta,
    )

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
    when the annotation expects ``str``.

    Args:
      param_name: The name of the parameter (for error messages).
      value: The value to coerce.
      annotated_type: The type annotation of the parameter.

    Returns:
      The coerced value.
    """
    # Content → str auto-conversion (e.g. user content from START node).
    if isinstance(value, types.Content) and _expects_str(annotated_type):
      return _content_to_str(value, self.name, param_name)
    adapter = self._type_adapters.get(param_name)
    if adapter is None:
      adapter = TypeAdapter(annotated_type)
    return adapter.validate_python(value)

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> FunctionNode:
    copied = cast(FunctionNode, super().model_copy(update=update, deep=deep))
    if not update or 'name' not in update:
      return copied

    # If the wrapped function is a bound method of a Node, we need to clone
    # the Node and re-bind the function to the new instance.
    # This is needed if the function is referring to params like 'name' from the "self" reference.
    # Like Workflow or LLM use that name for event node_paths or retreving session events.
    func = self._func
    if inspect.ismethod(func) and isinstance(
        getattr(func, '__self__', None), BaseNode
    ):
      method_self = getattr(func, '__self__')
      method_name = getattr(func, '__name__')

      # Pass the name update to the cloned agent instance if it's being passed
      # to the FunctionNode (case for parallel workers).
      agent_update = {
          'name': update['name'],
      }

      new_obj = method_self.model_copy(update=agent_update)
      copied._func = getattr(new_obj, method_name)
    else:
      copied._func = func

    copied._sig = self._sig
    copied._type_hints = self._type_hints
    copied._type_adapters = self._type_adapters
    copied._context_param_name = self._context_param_name
    return copied

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    # --- Auth gate ---
    if self.auth_config:
      interrupt_id = f'wf_auth:{ctx.node_path}'
      auth_response = ctx.resume_inputs.get(interrupt_id)
      if auth_response is not None:
        await process_auth_resume(auth_response, self.auth_config, ctx.state)
      elif not has_auth_credential(self.auth_config, ctx.state):
        yield create_auth_request_event(self.auth_config, interrupt_id)
        return

    kwargs = self._bind_parameters(ctx, node_input)

    unwrapped_func = _unwrap_callable(self._func)
    if inspect.isasyncgenfunction(unwrapped_func):
      items = self._func(**kwargs)
    elif inspect.isgeneratorfunction(unwrapped_func):
      items = _sync_to_async_gen(self._func(**kwargs))
    else:
      items = None

    if items is not None:
      async for item in items:
        event = self._to_event(ctx, item)
        if event is not None:
          yield event
    else:
      if inspect.iscoroutinefunction(unwrapped_func):
        result = await self._func(**kwargs)
      else:  # Sync function
        result = self._func(**kwargs)

      event = self._to_event(ctx, result)
      if event is not None:
        yield event
