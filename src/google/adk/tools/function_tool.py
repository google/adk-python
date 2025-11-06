# Copyright 2025 Google LLC
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

import inspect
import logging
from typing import Any
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Union

from google.genai import types
import pydantic
from typing_extensions import override

from ..utils.context_utils import Aclosing
from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


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
    name = ''
    doc = ''
    # Handle different types of callables
    if hasattr(func, '__name__'):
      # Regular functions, unbound methods, etc.
      name = func.__name__
    elif hasattr(func, '__class__'):
      # Callable objects, bound methods, etc.
      name = func.__class__.__name__

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
    self._ignore_params = ['tool_context', 'input_stream']
    self._require_confirmation = require_confirmation

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    function_decl = types.FunctionDeclaration.model_validate(
        build_function_declaration(
            func=self.func,
            # The model doesn't understand the function context.
            # input_stream is for streaming tool
            ignore_params=self._ignore_params,
            variant=self._api_variant,
        )
    )

    return function_decl

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

    for param_name, param in signature.parameters.items():
      if param_name in args and param.annotation != inspect.Parameter.empty:
        target_type = param.annotation

        # Handle Optional[PydanticModel] types
        if get_origin(param.annotation) is Union:
          union_args = get_args(param.annotation)
          # Find the non-None type in Optional[T] (which is Union[T, None])
          non_none_types = [arg for arg in union_args if arg is not type(None)]
          if len(non_none_types) == 1:
            target_type = non_none_types[0]

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
                  f"Failed to convert argument '{param_name}' to Pydantic model"
                  f' {target_type.__name__}: {e}'
              )
              # Keep the original value if conversion fails
              pass

    return converted_args

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    # Preprocess arguments (includes Pydantic model conversion)
    args_to_call = self._preprocess_args(args)

    signature = inspect.signature(self.func)
    valid_params = {param for param in signature.parameters}
    if 'tool_context' in valid_params:
      args_to_call['tool_context'] = tool_context

    # Filter args_to_call to only include valid parameters for the function
    args_to_call = {k: v for k, v in args_to_call.items() if k in valid_params}

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

    if isinstance(self._require_confirmation, Callable):
      require_confirmation = await self._invoke_callable(
          self._require_confirmation, args_to_call
      )
    else:
      require_confirmation = bool(self._require_confirmation)

    if require_confirmation:
      if not tool_context.tool_confirmation:
        args_to_show = args_to_call.copy()
        if 'tool_context' in args_to_show:
          args_to_show.pop('tool_context')

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
    else:
      return target(**args_to_call)

  # TODO(hangfei): fix call live for function stream.
  async def _call_live(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
      invocation_context,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if (
        self.name in invocation_context.active_streaming_tools
        and invocation_context.active_streaming_tools[self.name].stream
    ):
      args_to_call['input_stream'] = invocation_context.active_streaming_tools[
          self.name
      ].stream
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context

    # TODO: support tool confirmation for live mode.
    async with Aclosing(self.func(**args_to_call)) as agen:
      async for item in agen:
        yield item

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
      #
      # For more refer to: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
      if param.default == inspect.Parameter.empty and param.kind not in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        mandatory_params.append(name)

    return mandatory_params
