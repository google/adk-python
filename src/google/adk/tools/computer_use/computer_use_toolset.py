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
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import functools
import inspect
import logging
import math
import threading
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import Optional
from typing import Union
import weakref

from google.genai import types
from packaging.version import InvalidVersion
from packaging.version import Version
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...models.llm_request import LlmRequest
from ...utils.model_name_utils import extract_model_name
from .._url_validator import _is_blocked_hostname
from .._url_validator import _parse_request_target
from .._url_validator import _resolve_direct_addresses
from ..base_toolset import BaseToolset
from ..tool_context import ToolContext
from .base_computer import BaseComputer
from .base_computer import ComputerState
from .computer_use_tool import ComputerUseTool

# Methods that should be excluded when creating tools from BaseComputer methods
EXCLUDED_METHODS = {
    "screen_size",
    "environment",
    "close",
    "prepare",
}

_DIRECT_EXCLUSION_MAP: Mapping[str, frozenset[str]] = MappingProxyType({
    # Legacy -> Modern
    "click_at": frozenset({"click"}),
    "hover_at": frozenset({"move"}),
    "type_text_at": frozenset({"type"}),
    "type_text": frozenset({"type"}),
    "scroll_at": frozenset({"scroll"}),
    # Whole-page scroll has no modern counterpart
    "scroll_document": frozenset(),
    "current_state": frozenset({"take_screenshot"}),
    "key_combination": frozenset({"press_key", "hotkey"}),
    "wait_5_seconds": frozenset({"wait"}),
    # Modern -> Legacy
    "click": frozenset({"click_at"}),
    "move": frozenset({"hover_at"}),
    "type": frozenset({"type_text_at"}),
    "scroll": frozenset({"scroll_at"}),  # Does NOT drop scroll_document
    "take_screenshot": frozenset({"current_state"}),
    "press_key": frozenset(),  # Handled by conjunctive post-pass below
    "hotkey": frozenset(),  # Handled by conjunctive post-pass below
    "wait": frozenset({"wait_5_seconds"}),
})

_OPTIONAL_MOUSE_ACTIONS = (
    "double_click",
    "triple_click",
    "middle_click",
    "right_click",
    "mouse_down",
    "mouse_up",
)
_OPTIONAL_KEY_ACTIONS = (
    "key_down",
    "key_up",
)

_LEGACY_EXCLUSION_NAMES: frozenset[str] = frozenset({
    "click_at",
    "hover_at",
    "type_text_at",
    "type_text",
    "scroll_at",
    "scroll_document",
    "current_state",
    "key_combination",
    "wait_5_seconds",
})

_MODERN_EXCLUSION_NAMES: frozenset[str] = frozenset({
    "click",
    "move",
    "type",
    "scroll",
    "take_screenshot",
    "press_key",
    "hotkey",
    "wait",
    *_OPTIONAL_MOUSE_ACTIONS,
    *_OPTIONAL_KEY_ACTIONS,
})


def _expand_local_exclusions(excluded_names: Iterable[str]) -> set[str]:
  """Expands configured exclusions across legacy and modern tool names."""
  expanded = set(excluded_names)
  for name in list(expanded):
    expanded |= _DIRECT_EXCLUSION_MAP.get(name, frozenset())
  if {"type", "type_text_at", "type_text"} & expanded:
    expanded |= {"type", "type_text_at", "type_text"}
  # Conjunctive post-pass: only drop legacy key_combination if BOTH modern
  # keyboard primitives (press_key and hotkey) were excluded.
  if {"press_key", "hotkey"} <= expanded:
    expanded.add("key_combination")
  return expanded


def _parse_gemini_major_version(model_string: Optional[str]) -> Optional[int]:
  """Returns the major version of a `gemini-*` model, or None if unparseable."""
  if not model_string:
    return None
  model_name = extract_model_name(model_string)
  if not model_name.startswith("gemini-"):
    return None
  version_part = model_name[len("gemini-") :].split("-", 1)[0]
  try:
    return Version(version_part).major
  except InvalidVersion:
    return None


def _is_legacy_gemini_model(model_string: Optional[str]) -> bool:
  """Returns True if model_string is explicitly a Gemini 1.x or 2.x model.

  Args:
    model_string: The model name or resource path from the LLM request.
      Unparseable model identifiers (such as non-numeric aliases like
      `gemini-flash-latest`, EAP models, or custom/tuned Vertex AI endpoint
      resource paths) default to False (the modern Gemini 3.x action schema).
  """
  major = _parse_gemini_major_version(model_string)
  return major is not None and major < 3


def _resolve_excluded_predefined_functions_for_model(
    excluded_names: Optional[list[str]],
    model: Optional[str],
    unimplemented_modern_actions: Iterable[str] = (),
) -> Optional[list[str]]:
  """Translates excluded function names for the target model schema."""
  if _is_legacy_gemini_model(model):
    if excluded_names is None:
      return None
    resolved: list[str] = []
    excluded_set = set(excluded_names)
    both_modern_keys = {"press_key", "hotkey"} <= excluded_set
    for name in excluded_names:
      if name in ("press_key", "hotkey"):
        if both_modern_keys:
          resolved.append("key_combination")
      elif name == "type_text":
        resolved.append("type_text_at")
      elif name in _MODERN_EXCLUSION_NAMES:
        resolved.extend(sorted(_DIRECT_EXCLUSION_MAP.get(name, frozenset())))
      else:
        resolved.append(name)
    return list(dict.fromkeys(resolved))

  resolved = []
  for name in excluded_names or ():
    if name in _LEGACY_EXCLUSION_NAMES:
      mapped_names = (
          ("press_key", "hotkey")
          if name == "key_combination"
          else sorted(_DIRECT_EXCLUSION_MAP.get(name, frozenset()))
      )
      resolved.extend(mapped_names)
    else:
      resolved.append(name)
  major = _parse_gemini_major_version(model)
  if major is not None and major >= 3:
    resolved.extend(unimplemented_modern_actions)
  deduped = list(dict.fromkeys(resolved))
  if not deduped and excluded_names is None:
    return None
  return deduped


_URL_REFUSED_ERROR = (
    "navigate refused: url must be http(s) and must not target a private or"
    " link-local address."
)

# Per-(computer, event loop) locks serializing concurrent computer actions
# both within a single model response and across sessions sharing the same
# computer instance (preventing a concurrent session's `prepare()` from
# rebinding `AgentEngineSandboxComputer._session_state` mid-action and
# returning another session's screenshot). Keyed by `id(computer)` with a
# weakref cleanup callback so unhashable `BaseComputer` subclasses (such as a
# plain `@dataclass`) work without `TypeError`, `ComputerUseToolset` remains
# deepcopy- and cloudpickle-serializable, and locks stay bound to the running
# loop across separate `asyncio.run()` invocations.
_COMPUTER_EXECUTION_LOCKS: dict[
    int,
    tuple[
        Optional[weakref.ref[BaseComputer]],
        weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock],
    ],
] = {}
_COMPUTER_EXECUTION_LOCKS_GUARD = threading.RLock()


def _cleanup_computer_execution_locks(
    computer_id: int, dead_ref: weakref.ref[BaseComputer]
) -> None:
  """Removes per-loop locks when a `BaseComputer` instance is collected."""
  with _COMPUTER_EXECUTION_LOCKS_GUARD:
    entry = _COMPUTER_EXECUTION_LOCKS.get(computer_id)
    if entry is not None and entry[0] is dead_ref:
      del _COMPUTER_EXECUTION_LOCKS[computer_id]


def _get_computer_execution_lock(computer: BaseComputer) -> asyncio.Lock:
  """Returns the execution lock for `computer` on the current event loop."""
  loop = asyncio.get_running_loop()
  computer_id = id(computer)
  with _COMPUTER_EXECUTION_LOCKS_GUARD:
    entry = _COMPUTER_EXECUTION_LOCKS.get(computer_id)
    if entry is None or (entry[0] is not None and entry[0]() is not computer):
      loop_locks: weakref.WeakKeyDictionary[
          asyncio.AbstractEventLoop, asyncio.Lock
      ] = weakref.WeakKeyDictionary()
      try:
        comp_ref: Optional[weakref.ref[BaseComputer]] = weakref.ref(
            computer,
            functools.partial(_cleanup_computer_execution_locks, computer_id),
        )
      except TypeError:
        comp_ref = None
      _COMPUTER_EXECUTION_LOCKS[computer_id] = (comp_ref, loop_locks)
    else:
      loop_locks = entry[1]
      for stale_loop in [k for k in loop_locks if k.is_closed()]:
        del loop_locks[stale_loop]
    lock = loop_locks.get(loop)
    if lock is None:
      lock = asyncio.Lock()
      loop_locks[loop] = lock
    return lock


logger = logging.getLogger("google_adk." + __name__)


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
      excluded_predefined_functions: Names of computer use actions that should
        not be exposed as tools. Both legacy `BaseComputer` method names (e.g.
        `click_at`) and Gemini 3.x action names (e.g. `click`) are accepted;
        each is expanded to its counterpart locally and translated to the target
        model's action schema in the LLM request. Unparseable model identifiers
        (e.g. non-numeric aliases or custom endpoint resource paths) default to
        the Gemini 3.x action schema.
      allow_private_network_access: By default `navigate` refuses urls whose
        host is not publicly routable. Set this to True when the agent is meant
        to drive the browser against localhost or an internal host.
    """
    super().__init__()
    self._computer = computer
    self._excluded_predefined_functions = excluded_predefined_functions
    self._allow_private_network_access = allow_private_network_access
    self._initialized = False
    self._closed = False
    self._tools: Optional[list[ComputerUseTool]] = None
    self._init_lock_map: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
    self._lock_map_lock = threading.Lock()

  def _get_init_lock(self) -> asyncio.Lock:
    # Note: Mutual exclusion for lazy initialization is guaranteed per event
    # loop; concurrent threads running distinct event loops on a shared toolset
    # instance each acquire their own loop-local lock.
    current_loop = asyncio.get_running_loop()
    with self._lock_map_lock:
      lock = self._init_lock_map.get(current_loop)
      if lock is None:
        for stale_loop in [
            loop for loop in self._init_lock_map if loop.is_closed()
        ]:
          del self._init_lock_map[stale_loop]
        lock = asyncio.Lock()
        self._init_lock_map[current_loop] = lock
      return lock

  def __getstate__(self) -> dict[str, Any]:
    state = dict(super().__getstate__())
    state.pop("_lock_map_lock", None)
    state["_init_lock_map"] = {}
    state["_initialized"] = False
    state["_closed"] = False
    state["_tools"] = None
    state["_cached_prefixed_tools"] = None
    state["_cached_invocation_id"] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    super_setstate = getattr(super(), "__setstate__", None)
    if super_setstate is not None:
      super_setstate(state)
    else:
      self.__dict__.update(state)
    self._init_lock_map = {}
    self._lock_map_lock = threading.Lock()

  async def _ensure_initialized(self) -> None:
    if self._initialized:
      return
    async with self._get_init_lock():
      if not self._initialized:
        await self._computer.initialize()
        self._initialized = True
        self._closed = False

  def _is_implemented_on_computer(
      self,
      method_name: str,
      required_params: tuple[str, ...] = (),
      optional_params: tuple[str, ...] = (),
  ) -> bool:
    """Probes the computer class for a matching coroutine override."""
    attr = getattr(self._computer.__class__, method_name, None)
    if not inspect.iscoroutinefunction(attr):
      return False
    try:
      sig = inspect.signature(attr)
    except (TypeError, ValueError):
      return False
    ret = sig.return_annotation
    if ret is not inspect.Signature.empty and not (
        ret is ComputerState
        or (isinstance(ret, str) and ret.split(".")[-1] == "ComputerState")
    ):
      return False
    params = [p for p in sig.parameters.values() if p.name != "self"]
    param_by_name = {
        p.name: p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    for req in required_params:
      if req not in param_by_name and not has_var_kw:
        return False
    allowed_optional = set(optional_params) | {"intent", "tool_context"}
    for p in params:
      if p.kind in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        continue
      if p.kind == inspect.Parameter.POSITIONAL_ONLY:
        return False
      if p.name in required_params:
        continue
      if p.default is inspect.Parameter.empty or p.name not in allowed_optional:
        return False
    return True

  def _unimplemented_modern_actions(self) -> list[str]:
    """Returns 3.x actions that this computer does not implement."""
    unimplemented: list[str] = []
    for action_name in _OPTIONAL_MOUSE_ACTIONS:
      if not self._is_implemented_on_computer(
          action_name,
          required_params=("x", "y"),
          optional_params=("button",),
      ):
        unimplemented.append(action_name)
    for action_name in _OPTIONAL_KEY_ACTIONS:
      if not self._is_implemented_on_computer(
          action_name, required_params=("key",)
      ):
        unimplemented.append(action_name)
    return unimplemented

  def _wrap_method_with_state_binding(
      self,
      method: Callable[..., Any],
      *,
      tool_name: Optional[str] = None,
  ) -> Callable[..., Any]:
    """Wrap a computer method to bind session state from tool_context.

    This wrapper intercepts the tool_context parameter injected by ADK's
    runtime, serializes concurrent actions against the same computer on the
    running event loop (both within a single model response and across sessions
    sharing the computer), and prepares the computer before calling the actual
    method. tool_context is forwarded to the wrapped callable only when its
    signature declares a tool_context parameter.

    Args:
      method: The computer method or fresh shim to wrap.
      tool_name: Optional explicit tool name to assign to the wrapper.

    Returns:
      A wrapped method that binds session state around calling the method.
    """
    computer = self._computer
    resolved_name = tool_name or method.__name__
    try:
      orig_sig = inspect.signature(method)
    except TypeError:
      orig_sig = inspect.Signature([
          inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
          inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
      ])
    forward_tool_context = (
        "tool_context" in orig_sig.parameters
        and orig_sig.parameters["tool_context"].kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )

    @functools.wraps(method)
    async def wrapper(
        *args: Any,
        tool_context: Optional[ToolContext] = None,
        **kwargs: Any,
    ) -> Any:
      # Serialize concurrent computer actions on this computer (both when
      # Gemini 3.x returns multiple actions in one response and when multiple
      # sessions share a computer such as `AgentEngineSandboxComputer`), so
      # each action's `prepare()` and execution complete before another action
      # rebinds computer state or takes a screenshot.
      async with _get_computer_execution_lock(computer):
        # Prepare computer before each tool call
        # Computers that need session state (e.g., AgentEngineSandboxComputer)
        # override prepare() to bind state for sandbox/token sharing
        if tool_context is not None:
          await computer.prepare(tool_context)

        if forward_tool_context:
          return await method(*args, tool_context=tool_context, **kwargs)
        # Call the original method (without tool_context - computer doesn't
        # need it)
        return await method(*args, **kwargs)

    wrapper.__name__ = resolved_name
    wrapper.__qualname__ = resolved_name

    # Create signature including both original parameters and tool_context.
    # FunctionTool filters args based on signature params.
    non_var_kw_params = [
        p
        for p in orig_sig.parameters.values()
        if p.name != "tool_context" and p.kind != inspect.Parameter.VAR_KEYWORD
    ]
    var_kw_params = [
        p
        for p in orig_sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    new_params = (
        non_var_kw_params
        + [
            inspect.Parameter(
                "tool_context",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=ToolContext,
            )
        ]
        + var_kw_params
    )
    setattr(wrapper, "__signature__", orig_sig.replace(parameters=new_params))

    return wrapper

  def _wrap_navigate_with_url_validation(
      self, navigate_method: Callable[..., Any]
  ) -> Callable[..., Any]:
    """Checks a model-supplied url before `navigate` hands it to the browser."""

    @functools.wraps(navigate_method)
    async def wrapper(url: str, *args: Any, **kwargs: Any) -> Any:
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
      return await navigate_method(url, *args, **kwargs)

    return wrapper

  def _build_modern_alias_methods(
      self, excluded_set: Collection[str]
  ) -> list[Callable[..., Any]]:
    """Builds modern Gemini 3.x shims that delegate to BaseComputer."""
    modern_methods: list[Callable[..., Any]] = []

    if "click" not in excluded_set:
      if self._is_implemented_on_computer(
          "click", required_params=("x", "y"), optional_params=("button",)
      ):
        click_fn = getattr(self._computer, "click")
      else:

        async def click(x: int, y: int) -> ComputerState:
          """Clicks at the specified (x, y) coordinates."""
          return await self._computer.click_at(x, y)

        click_fn = click
      modern_methods.append(
          self._wrap_method_with_state_binding(click_fn, tool_name="click")
      )

    if "move" not in excluded_set:
      if self._is_implemented_on_computer("move", required_params=("x", "y")):
        move_fn = getattr(self._computer, "move")
      else:

        async def move(x: int, y: int) -> ComputerState:
          """Moves the cursor to the specified (x, y) coordinates."""
          return await self._computer.hover_at(x, y)

        move_fn = move
      modern_methods.append(
          self._wrap_method_with_state_binding(move_fn, tool_name="move")
      )

    if "type" not in excluded_set:
      if self._is_implemented_on_computer(
          "type", required_params=("text",), optional_params=("press_enter",)
      ):
        type_fn = getattr(self._computer, "type")
      else:

        async def type_action(
            text: str, press_enter: bool = False
        ) -> ComputerState:
          """Types text into the currently focused element."""
          return await self._computer.type_text(
              text=text, press_enter=press_enter
          )

        type_fn = type_action
      modern_methods.append(
          self._wrap_method_with_state_binding(type_fn, tool_name="type")
      )

    if "scroll" not in excluded_set:
      if self._is_implemented_on_computer(
          "scroll",
          required_params=("x", "y", "direction"),
          optional_params=("magnitude_in_pixels",),
      ):
        scroll_fn = getattr(self._computer, "scroll")
      else:

        async def scroll(
            x: int,
            y: int,
            direction: Literal["up", "down", "left", "right"] = "down",
            magnitude_in_pixels: int = 300,
        ) -> ComputerState:
          """Scrolls the webpage at the specified coordinates."""
          return await self._computer.scroll_at(
              x, y, direction, magnitude_in_pixels
          )

        scroll_fn = scroll
      modern_methods.append(
          self._wrap_method_with_state_binding(scroll_fn, tool_name="scroll")
      )

    if "take_screenshot" not in excluded_set:
      if self._is_implemented_on_computer("take_screenshot"):
        screenshot_fn = getattr(self._computer, "take_screenshot")
      else:

        async def take_screenshot() -> ComputerState:
          """Takes a screenshot of the current webpage state."""
          return await self._computer.current_state()

        screenshot_fn = take_screenshot
      modern_methods.append(
          self._wrap_method_with_state_binding(
              screenshot_fn, tool_name="take_screenshot"
          )
      )

    if "press_key" not in excluded_set:
      if self._is_implemented_on_computer(
          "press_key", required_params=("key",)
      ):
        press_key_fn = getattr(self._computer, "press_key")
      else:

        async def press_key(key: str) -> ComputerState:
          """Presses a single keyboard key."""
          return await self._computer.key_combination([key])

        press_key_fn = press_key
      modern_methods.append(
          self._wrap_method_with_state_binding(
              press_key_fn, tool_name="press_key"
          )
      )

    if "hotkey" not in excluded_set:
      if self._is_implemented_on_computer("hotkey", required_params=("keys",)):
        hotkey_fn = getattr(self._computer, "hotkey")
      else:

        async def hotkey(keys: list[str]) -> ComputerState:
          """Presses a combination of keyboard keys simultaneously."""
          return await self._computer.key_combination(keys)

        hotkey_fn = hotkey
      modern_methods.append(
          self._wrap_method_with_state_binding(hotkey_fn, tool_name="hotkey")
      )

    for action_name in _OPTIONAL_MOUSE_ACTIONS:
      if action_name in excluded_set:
        continue
      if self._is_implemented_on_computer(
          action_name, required_params=("x", "y"), optional_params=("button",)
      ):
        fn = getattr(self._computer, action_name)
        modern_methods.append(
            self._wrap_method_with_state_binding(fn, tool_name=action_name)
        )

    for action_name in _OPTIONAL_KEY_ACTIONS:
      if action_name in excluded_set:
        continue
      if self._is_implemented_on_computer(
          action_name, required_params=("key",)
      ):
        fn = getattr(self._computer, action_name)
        modern_methods.append(
            self._wrap_method_with_state_binding(fn, tool_name=action_name)
        )

    return modern_methods

  @staticmethod
  async def adapt_computer_use_tool(
      method_name: str,
      adapter_func: Union[
          Callable[[Callable[..., Any]], Callable[..., Any]],
          Callable[[Callable[..., Any]], Any],
      ],
      llm_request: LlmRequest,
  ) -> None:
    """Adapts a computer use tool by replacing it with a modified version.

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
    if method_name in EXCLUDED_METHODS or method_name == "type_text":
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

    original_tool = cast(ComputerUseTool, llm_request.tools_dict[method_name])

    # Create the adapted function using the adapter
    adapted_func_or_awaitable = adapter_func(original_tool.func)
    if inspect.isawaitable(adapted_func_or_awaitable):
      adapted_func = await adapted_func_or_awaitable
    else:
      adapted_func = adapted_func_or_awaitable

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
    if new_method_name != method_name:
      del llm_request.tools_dict[method_name]

    logger.debug(
        "Adapted tool %s to %s with adapter function",
        method_name,
        new_method_name,
    )

  @override
  # list is invariant, so the narrower element type is not a compatible
  # override; widening it to BaseTool would change this public signature.
  async def get_tools(  # type: ignore[override]
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[ComputerUseTool]:
    if self._tools:
      return self._tools
    await self._ensure_initialized()
    async with self._get_init_lock():
      if self._tools:
        return self._tools
      # Get screen size for tool configuration
      screen_size = await self._computer.screen_size()

      excluded_set = _expand_local_exclusions(
          self._excluded_predefined_functions or ()
      )

      # Get methods defined in Computer ABC, excluding specified methods
      computer_methods = []

      # Get all methods defined in the Computer ABC interface
      for method_name in dir(BaseComputer):
        # Skip private methods (starting with underscore)
        if method_name.startswith("_"):
          continue

        # Skip excluded methods
        if method_name in EXCLUDED_METHODS:
          continue

        # Skip session_state property and coordinate-free type_text helper
        # (invoked through the modern `type` tool shim, not exposed directly)
        if method_name in ("session_state", "type_text"):
          continue

        # Skip methods excluded by configuration
        if method_name in excluded_set:
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
          elif method_name == "wait":
            raw_wait_method = instance_method

            async def wait(seconds: float = 1) -> ComputerState:
              """Waits for the specified number of seconds."""
              return cast(
                  ComputerState,
                  await raw_wait_method(max(0, math.ceil(seconds))),
              )

            instance_method = wait
          # Wrap with state binding so session_state is set before each call
          wrapped_method = self._wrap_method_with_state_binding(
              instance_method, tool_name=method_name
          )
          computer_methods.append(wrapped_method)

      # Append modern Gemini 3.x shims that translate to BaseComputer methods
      computer_methods.extend(self._build_modern_alias_methods(excluded_set))

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
    async with self._get_init_lock():
      if not self._closed:
        await self._computer.close()
        self._closed = True
      self._initialized = False
      self._tools = None
      self._cached_prefixed_tools = None
      self._cached_invocation_id = None
    await super().close()

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Adds tools and computer use configuration to the LLM request."""
    try:

      # Add this tool to the tools dictionary
      if not self._tools:
        await self.get_tools()

      assert self._tools is not None
      for computer_tool in self._tools:
        llm_request.tools_dict[computer_tool.name] = computer_tool

      # Initialize config if needed
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []

      # Check if computer use is already configured
      for configured_tool in llm_request.config.tools:
        if (
            isinstance(configured_tool, types.Tool)
            and configured_tool.computer_use
        ):
          logger.debug("Computer use already configured in LLM request")
          return

      # Add computer use tool configuration
      computer_environment = await self._computer.environment()
      environment = getattr(
          types.Environment,
          computer_environment.name,
          types.Environment.ENVIRONMENT_BROWSER,
      )
      resolved_excluded_functions = (
          _resolve_excluded_predefined_functions_for_model(
              self._excluded_predefined_functions,
              llm_request.model,
              unimplemented_modern_actions=self._unimplemented_modern_actions(),
          )
      )
      llm_request.config.tools.append(
          types.Tool(
              computer_use=types.ComputerUse(
                  environment=environment,
                  excluded_predefined_functions=resolved_excluded_functions,
              )
          )
      )
      logger.debug(
          "Added computer use tool with environment: %s,"
          " excluded_functions: %s",
          environment,
          resolved_excluded_functions,
      )

    except Exception as e:
      logger.error("Error in ComputerUseToolset.process_llm_request: %s", e)
      raise
