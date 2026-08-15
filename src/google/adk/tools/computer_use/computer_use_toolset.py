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
import functools
import inspect
import logging
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

from google.genai import types
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...models.llm_request import LlmRequest
from ..base_toolset import BaseToolset
from ..tool_context import ToolContext
from .base_computer import BaseComputer
from .base_computer import ComputerState
from .computer_use_tool import ComputerUseTool

# Methods that should be excluded when creating tools from BaseComputer methods
EXCLUDED_METHODS = {"screen_size", "environment", "close", "prepare"}

_URL_REFUSED_ERROR = (
    "navigate refused: url must be http(s) and must not target a private or"
    " link-local address."
)

logger = logging.getLogger("google_adk." + __name__)


@experimental(FeatureName.COMPUTER_USE)
class ComputerUseToolset(BaseToolset):

  def __init__(
      self,
      *,
      computer: BaseComputer,
      excluded_predefined_functions: Optional[list[str]] = None,
      allow_private_network_access: bool = False,
  ):
    """Initializes the ComputerUseToolset.

    Args:
      computer: The computer environment to expose as tools.
      excluded_predefined_functions: Names of BaseComputer methods that should
        not be exposed as tools.
      allow_private_network_access: By default `navigate` refuses urls whose
        host is not publicly routable. Set this to True when the agent is
        meant to drive the browser against localhost or an internal host.
    """
    super().__init__()
    self._computer = computer
    self._excluded_predefined_functions = excluded_predefined_functions
    self._allow_private_network_access = allow_private_network_access
    self._initialized = False
    self._tools = None

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self._computer.initialize()
      self._initialized = True

  def _wrap_method_with_state_binding(
      self, method: Callable[..., Any]
  ) -> Callable[..., Any]:
    """Wrap a computer method to bind session state from tool_context.

    This wrapper intercepts the tool_context parameter injected by ADK's
    runtime and binds it to the computer's session_state property before
    calling the actual method. This allows computers to access session
    state without being coupled to tool_context directly.

    Args:
      method: The computer method to wrap.

    Returns:
      A wrapped method that binds session state before calling.
    """
    computer = self._computer

    @functools.wraps(method)
    async def wrapper(
        *args: Any, tool_context: ToolContext = None, **kwargs: Any
    ) -> Any:
      # Prepare computer before each tool call
      # Computers that need session state (e.g., AgentEngineSandboxComputer)
      # override prepare() to bind state for sandbox/token sharing
      if tool_context is not None:
        await computer.prepare(tool_context)

      # Call the original method (without tool_context - computer doesn't need it)
      return await method(*args, **kwargs)

    # Create a signature that includes both original parameters and tool_context.
    # This is needed because FunctionTool filters args based on signature params.
    orig_sig = inspect.signature(method)
    new_params = list(orig_sig.parameters.values()) + [
        inspect.Parameter(
            "tool_context",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=ToolContext,
        )
    ]
    wrapper.__signature__ = orig_sig.replace(parameters=new_params)

    return wrapper

  def _wrap_navigate_with_url_validation(
      self, navigate_method: Callable[..., Any]
  ) -> Callable[..., Any]:
    """Checks a model-supplied url before `navigate` hands it to the browser."""

    @functools.wraps(navigate_method)
    async def wrapper(url: str) -> Any:
      # Deferred to keep `requests` off the computer-use import path.
      from ..load_web_page import _is_blocked_hostname
      from ..load_web_page import _parse_request_target
      from ..load_web_page import _resolve_direct_addresses

      try:
        if not isinstance(url, str):
          raise ValueError("url is not a string")
        target = _parse_request_target(url)
        # A browser ends the authority at "\" but urlparse does not: in
        # `http://169.254.169.254\@example.com/` the host is example.com here
        # and 169.254.169.254 in Chrome, so refuse instead of checking it.
        if "\\" in target.parsed_url.netloc:
          raise ValueError("backslash in hostname")
        if not self._allow_private_network_access:
          if _is_blocked_hostname(target.hostname):
            raise ValueError("hostname is blocked")
          # getaddrinfo blocks, so keep it off the event loop.
          await asyncio.to_thread(_resolve_direct_addresses, target.hostname)
      except ValueError:
        logger.warning("Refusing navigate(): url failed safety validation.")
        # The computer-use model rejects a function response with no url,
        # so report the page the browser is currently on.
        state: ComputerState = await self._computer.current_state()
        return {"error": _URL_REFUSED_ERROR, "url": state.url}
      return await navigate_method(url)

    return wrapper

  @staticmethod
  async def adapt_computer_use_tool(
      method_name: str,
      adapter_func: Union[
          Callable[[Callable[..., Any]], Callable[..., Any]],
          Callable[[Callable[..., Any]], Any],
      ],
      llm_request: LlmRequest,
  ) -> None:
    """Adapt a computer use tool by replacing it with a modified version.

    Args:
      method_name: The name of the method (of BaseComputer class) to adapt (e.g.
        'wait').
      adapter_func: A function that accepts existing computer use async function
        and returns a new computer use async function. Can be either sync or
        async function. The name of the returned function will be used as the
        new tool name.
      llm_request: The LLM request containing the tools dictionary.
    """
    # Validate that the method is a valid BaseComputer method
    if method_name in EXCLUDED_METHODS:
      logger.warning(
          "Method %s is not a valid BaseComputer method", method_name
      )
      return

    # Check if it's a method defined in BaseComputer class
    attr = getattr(BaseComputer, method_name, None)
    if attr is None or not callable(attr):
      logger.warning(
          "Method %s is not a valid BaseComputer method", method_name
      )
      return

    if method_name not in llm_request.tools_dict:
      logger.warning("Method %s not found in tools_dict", method_name)
      return

    original_tool = llm_request.tools_dict[method_name]

    # Create the adapted function using the adapter
    # Handle both sync and async adapter functions
    if asyncio.iscoroutinefunction(adapter_func):
      # If adapter_func is async, await it to get the adapted function
      adapted_func = await adapter_func(original_tool.func)
    else:
      # If adapter_func is sync, call it directly
      adapted_func = adapter_func(original_tool.func)

    # Get the name from the adapted function
    new_method_name = adapted_func.__name__

    # Create a new ComputerUseTool with the adapted function
    adapted_tool = ComputerUseTool(
        func=adapted_func,
        screen_size=original_tool._screen_size,
        virtual_screen_size=original_tool._coordinate_space,
    )

    # Add the adapted tool and remove the original
    llm_request.tools_dict[new_method_name] = adapted_tool
    del llm_request.tools_dict[method_name]

    logger.debug(
        "Adapted tool %s to %s with adapter function",
        method_name,
        new_method_name,
    )

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[ComputerUseTool]:
    if self._tools:
      return self._tools
    await self._ensure_initialized()
    # Get screen size for tool configuration
    screen_size = await self._computer.screen_size()

    # Get all methods defined in Computer abstract base class, excluding specified methods
    computer_methods = []

    # Get all methods defined in the Computer ABC interface
    for method_name in dir(BaseComputer):
      # Skip private methods (starting with underscore)
      if method_name.startswith("_"):
        continue

      # Skip excluded methods
      if method_name in EXCLUDED_METHODS:
        continue

      # Skip session_state property
      if method_name == "session_state":
        continue

      # Skip methods excluded by configuration
      if (
          self._excluded_predefined_functions
          and method_name in self._excluded_predefined_functions
      ):
        continue

      # Check if it's a method defined in Computer class
      attr = getattr(BaseComputer, method_name, None)
      if attr is not None and callable(attr):
        # Get the corresponding method from the concrete instance
        instance_method = getattr(self._computer, method_name)
        if method_name == "navigate":
          # Check the url the model supplied before it reaches the browser.
          instance_method = self._wrap_navigate_with_url_validation(
              instance_method
          )
        # Wrap with state binding so session_state is set before each call
        wrapped_method = self._wrap_method_with_state_binding(instance_method)
        computer_methods.append(wrapped_method)

    # Create ComputerUseTool instances for each wrapped method

    self._tools = [
        ComputerUseTool(
            func=method,
            screen_size=screen_size,
        )
        for method in computer_methods
    ]
    return self._tools

  @override
  async def close(self) -> None:
    await self._computer.close()

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Add its tools to the LLM request and add computer use configuration to the LLM request."""
    try:

      # Add this tool to the tools dictionary
      if not self._tools:
        await self.get_tools()

      for tool in self._tools:
        llm_request.tools_dict[tool.name] = tool

      # Initialize config if needed
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []

      # Check if computer use is already configured
      for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool) and tool.computer_use:
          logger.debug("Computer use already configured in LLM request")
          return

      # Add computer use tool configuration
      computer_environment = await self._computer.environment()
      environment = getattr(
          types.Environment,
          computer_environment.name,
          types.Environment.ENVIRONMENT_BROWSER,
      )
      llm_request.config.tools.append(
          types.Tool(
              computer_use=types.ComputerUse(
                  environment=environment,
                  excluded_predefined_functions=self._excluded_predefined_functions,
              )
          )
      )
      logger.debug(
          "Added computer use tool with environment: %s,"
          " excluded_functions: %s",
          environment,
          self._excluded_predefined_functions,
      )

    except Exception as e:
      logger.error("Error in ComputerUseToolset.process_llm_request: %s", e)
      raise
