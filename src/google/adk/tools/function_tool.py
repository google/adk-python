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

from contextlib import contextmanager
import contextvars
import functools
import inspect
import logging
from types import UnionType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Iterator
from typing import Optional
from typing import Union

from google.genai import types
import pydantic
from typing_extensions import override

from . import _function_tool_declarations
from ..features import FeatureName
from ..features import is_feature_enabled
from ..utils._schema_utils import get_list_inner_type
from ..utils._schema_utils import is_list_of_basemodel
from ..utils.context_utils import find_context_parameter
from ..utils.variant_utils import GoogleLLMVariant
from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)

_SyncCallableRunner = Callable[
    [Callable[..., Any], dict[str, Any]], Awaitable[Any]
]
_SYNC_CALLABLE_RUNNER: contextvars.ContextVar[_SyncCallableRunner | None] = (
    contextvars.ContextVar('adk_sync_callable_runner', default=None)
)


@contextmanager
def _use_sync_callable_runner(
    runner: _SyncCallableRunner | None = None,
) -> Iterator[None]:
  """Binds the runner used for synchronous callables.

  Passing ``None`` clears the binding, which stops a worker-owned nested call
  from reusing the caller's runner.
  """
  token = _SYNC_CALLABLE_RUNNER.set(runner)
  try:
    yield
  finally:
    _SYNC_CALLABLE_RUNNER.reset(token)


@functools.lru_cache(maxsize=1024)
def _build_declaration_cached(
    func: Callable[..., Any],
    ignore_params: tuple[str, ...],
    variant: GoogleLLMVariant,
    json_schema_enabled: bool,
) -> types.FunctionDeclaration:
  """Builds (and caches) a tool's FunctionDeclaration.

  The build runs pydantic ``create_model`` + JSON-schema generation, which is
  expensive and otherwise re-run for every tool on every LLM call even though
  the result depends only on these (static) inputs. ``json_schema_enabled`` is
  part of the key so toggling the feature flag rebuilds.
  """
  del json_schema_enabled  # Only participates in the cache key.
  return types.FunctionDeclaration.model_validate(
      build_function_declaration(
          func=func,
          ignore_params=list(ignore_params),
          variant=variant,
      )
  )


class FunctionTool(BaseTool):
  """A tool that wraps a user-defined Python function.

  Attributes:
    func: The function to wrap.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      require_confirmation: Union[bool, Callable[..., bool]] = False,
  ):
    """Initializes the FunctionTool. Extracts metadata from a callable object.

    Args:
      func: The function to wrap.
      require_confirmation: Whether this tool requires confirmation. A boolean or
        a callable that takes the function's arguments and returns a boolean. If
        the callable returns True, the tool will require confirmation from the
        user.
    """
    doc = ''
    # Shared with the declaration builder so the name advertised to the model
    # and the name the tool is registered under cannot drift apart.
    name = _function_tool_declarations.get_callable_name(func)

    # Get documentation (prioritize direct __doc__ if available)
    if hasattr(func, '__doc__') and func.__doc__:
      doc = inspect.cleandoc(func.__doc__)
    elif (
        hasattr(func, '__call__')
        and hasattr(func.__call__, '__doc__')
        and func.__call__.__doc__
    ):
      # For callable objects, try to get docstring from __call__ method
      doc = inspect.cleandoc(func.__call__.__doc__)

    super().__init__(name=name, description=doc)
    self.func = func
    # Detect context parameter by type annotation, fallback to 'tool_context' name
    self._context_param_name = find_context_parameter(func) or 'tool_context'
    self._ignore_params = [self._context_param_name, 'input_stream']
    self._require_confirmation = require_confirmation

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    # `ignore_params` drops the function context and input_stream (for streaming
    # tools), which the model doesn't understand. Return a copy: the cached
    # declaration is shared and callers (e.g. toolset prefixing) mutate it.
    declaration = _build_declaration_cached(
        self.func,
        tuple(self._ignore_params),
        self._api_variant,
        is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL),
    )
    return declaration.model_copy(deep=True)

  def _preprocess_args(self, args: dict[str, Any]) -> dict[str, Any]:
    """Preprocess and convert function arguments before invocation.

    Currently handles:
    - Converting JSON dictionaries to Pydantic model instances where expected

    Future extensions could include:
    - Type coercion for other complex types
    - Validation and sanitization
    - Custom conversion logic

    Args:
      args: Raw arguments from the LLM tool call

    Returns:
      Processed arguments ready for function invocation
    """
    signature = inspect.signature(self.func)
    converted_args = args.copy()
    try:
      type_hints = get_type_hints(self.func)
    except (TypeError, NameError):
      # NameError: unresolved forward refs (e.g. recursive type aliases).
      # TypeError: non-function callables.
      if hasattr(self.func, '__call__'):
        try:
          type_hints = get_type_hints(self.func.__call__)
        except (TypeError, NameError):
          type_hints = {}
      else:
        type_hints = {}

    for param_name, param in signature.parameters.items():
      if param_name in args:
        target_type = type_hints.get(param_name, param.annotation)
        if target_type != inspect.Parameter.empty:

          # Handle Optional/Union types (e.g. Optional[PydanticModel], PydanticModel | None)
          origin = get_origin(target_type)
          if origin is Union or origin is UnionType:
            union_args = get_args(target_type)
            # Find the non-None type in Optional[T] (which is Union[T, None])
            non_none_types = [
                arg for arg in union_args if arg is not type(None)
            ]
            if len(non_none_types) == 1:
              target_type = non_none_types[0]
            elif len(non_none_types) > 1 and all(
                inspect.isclass(t) and issubclass(t, pydantic.BaseModel)
                for t in non_none_types
            ):
              if args[param_name] is None or isinstance(
                  args[param_name], tuple(non_none_types)
              ):
                continue
              try:
                converted_args[param_name] = pydantic.TypeAdapter(
                    target_type
                ).validate_python(args[param_name])
              except Exception as e:
                logger.warning(
                    f"Failed to convert argument '{param_name}' to"
                    f' {target_type}: {e}'
                )
              continue

          # Some session stores persist call args as a proto `Struct`, whose
          # only number type is double, so an int comes back as a float.
          if target_type is int and type(args[param_name]) is float:
            if args[param_name].is_integer():
              converted_args[param_name] = int(args[param_name])
            else:
              logger.warning(
                  "Argument '%s' is typed int but got non-integral %r;"
                  ' passing it through unchanged.',
                  param_name,
                  args[param_name],
              )
            continue

          # Check if the target type is a Pydantic model
          if inspect.isclass(target_type) and issubclass(
              target_type, pydantic.BaseModel
          ):
            # Skip conversion if the value is None and the parameter is Optional
            if args[param_name] is None:
              continue

            # Convert to Pydantic model if it's not already the correct type
            if not isinstance(args[param_name], target_type):
              try:
                converted_args[param_name] = target_type.model_validate(
                    args[param_name]
                )
              except Exception as e:
                logger.warning(
                    f"Failed to convert argument '{param_name}' to Pydantic"
                    f' model {target_type.__name__}: {e}'
                )
                # Keep the original value if conversion fails
                pass
          # Handle list[BaseModel] types
          elif is_list_of_basemodel(target_type) and isinstance(
              args[param_name], list
          ):
            item_type = get_list_inner_type(target_type)
            if item_type is not None:
              try:
                converted_args[param_name] = [
                    item_type.model_validate(item)
                    if isinstance(item, dict)
                    else item
                    for item in args[param_name]
                ]
              except Exception as e:
                logger.warning(
                    f"Failed to convert argument '{param_name}' to"
                    f' list[{item_type.__name__}]: {e}'
                )
                pass

    return converted_args

  def _prepare_invocation_args(
      self, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    """Prepare args for function invocation (preprocesses, injects context and filters)."""
    args_to_call = self._preprocess_args(args)
    signature = inspect.signature(self.func)
    valid_params = set(signature.parameters.keys())
    if self._context_param_name in valid_params:
      args_to_call[self._context_param_name] = tool_context
    # In live mode (bidirectional streaming), tools may accept an 'input_stream'
    # parameter (e.g., LiveRequestQueue) to receive real-time streaming data.
    # When registered in _process_function_live_helper, the framework attaches
    # the dedicated stream to invocation_context.active_streaming_tools[name].
    # If the tool signature expects 'input_stream', we inject that active stream.
    if 'input_stream' in valid_params:
      active_tools = tool_context._invocation_context.active_streaming_tools
      if (
          active_tools is not None
          and self.name in active_tools
          and active_tools[self.name].stream is not None
      ):
        args_to_call['input_stream'] = active_tools[self.name].stream
    return {k: v for k, v in args_to_call.items() if k in valid_params}

  @override
  async def check_require_confirmation(
      self, args: dict[str, Any], tool_context: ToolContext
  ) -> bool:
    if callable(self._require_confirmation):
      args_to_call = self._prepare_invocation_args(args, tool_context)
      return cast(
          bool,
          await self._invoke_callable(self._require_confirmation, args_to_call),
      )
    return bool(self._require_confirmation)

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    # Preprocess arguments (includes Pydantic model conversion)
    args_to_call = self._prepare_invocation_args(args, tool_context)

    # Before invoking the function, we check for if the list of args passed in
    # has all the mandatory arguments or not.
    # If the check fails, then we don't invoke the tool and let the Agent know
    # that there was a missing input parameter. This will basically help
    # the underlying model fix the issue and retry.
    mandatory_args = self._get_mandatory_args()
    missing_mandatory_args = [
        arg for arg in mandatory_args if arg not in args_to_call
    ]

    if missing_mandatory_args:
      missing_mandatory_args_str = '\n'.join(missing_mandatory_args)
      error_str = f"""Invoking `{self.name}()` failed as the following mandatory input parameters are not present:
{missing_mandatory_args_str}
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      return {'error': error_str}

    require_confirmation = await self.check_require_confirmation(
        args, tool_context
    )

    if require_confirmation:
      if not tool_context.tool_confirmation:
        args_to_show = args_to_call.copy()
        if self._context_param_name in args_to_show:
          args_to_show.pop(self._context_param_name)

        tool_context.request_confirmation(
            hint=(
                f'Please approve or reject the tool call {self.name}() by'
                ' responding with a FunctionResponse with an expected'
                ' ToolConfirmation payload.'
            ),
        )
        tool_context.actions.skip_summarization = True
        return {
            'error': (
                'This tool call requires confirmation, please approve or'
                ' reject.'
            )
        }
      elif not tool_context.tool_confirmation.confirmed:
        return {'error': 'This tool call is rejected.'}

    return await self._invoke_callable(self.func, args_to_call)

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get('error'):
      return 'TOOL_ERROR'
    return None

  async def _invoke_callable(
      self, target: Callable[..., Any], args_to_call: dict[str, Any]
  ) -> Any:
    """Invokes a callable, handling both sync and async cases."""

    # Functions are callable objects, but not all callable objects are functions
    # checking coroutine function is not enough. We also need to check whether
    # Callable's __call__ function is a coroutine function
    is_async = inspect.iscoroutinefunction(target) or (
        hasattr(target, '__call__')
        and inspect.iscoroutinefunction(target.__call__)
    )
    if is_async:
      return await target(**args_to_call)
    runner = _SYNC_CALLABLE_RUNNER.get()
    if runner is not None:
      return await runner(target, args_to_call)
    return target(**args_to_call)

  def _get_mandatory_args(
      self,
  ) -> list[str]:
    """Identifies mandatory parameters (those without default values) for a function.

    Returns:
      A list of strings, where each string is the name of a mandatory parameter.
    """
    signature = inspect.signature(self.func)
    mandatory_params = []

    for name, param in signature.parameters.items():
      # A parameter is mandatory if:
      # 1. It has no default value (param.default is inspect.Parameter.empty)
      # 2. It's not a variable positional (*args) or variable keyword (**kwargs) parameter
      # 3. It's not an internal parameter to ignore (e.g. tool_context, input_stream)
      #
      # For more refer to: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
      if (
          param.default == inspect.Parameter.empty
          and name not in self._ignore_params
          and param.kind
          not in (
              inspect.Parameter.VAR_POSITIONAL,
              inspect.Parameter.VAR_KEYWORD,
          )
      ):
        mandatory_params.append(name)

    return mandatory_params
