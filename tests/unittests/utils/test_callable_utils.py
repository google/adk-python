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

"""Unit tests for _callable_utils module."""

from __future__ import annotations

import functools
from typing import Optional

from google.adk.agents.context import Context
from google.adk.tools.tool_context import ToolContext
from google.adk.utils._callable_utils import CallableSpec
from google.adk.utils._callable_utils import get_type_hints_cached
from google.adk.utils._callable_utils import unwrap_callable
from google.adk.utils.context_utils import _is_context_type as is_context_type
from google.adk.utils.context_utils import find_context_parameter
import pydantic
import pytest


class SimpleModel(pydantic.BaseModel):
  value: int
  name: str = "default"


class OtherModel(pydantic.BaseModel):
  tag: str


class CallableClass:
  """Docstring on class."""

  def __call__(self, x: int) -> int:
    """Docstring on call method."""
    return x * 2


def test_unwrap_callable_plain_function():
  """Plain functions unwrap to themselves."""

  def sample():
    pass

  assert unwrap_callable(sample) is sample


def test_unwrap_callable_partial_and_methods():
  """Unwraps functools.partial and bound methods to underlying functions."""

  def original(a, b):
    return a + b

  p = functools.partial(original, 1)
  assert unwrap_callable(p) is original

  callable_obj = CallableClass()
  unwrapped = unwrap_callable(callable_obj)
  assert unwrapped == CallableClass.__call__


def test_unwrap_callable_builtins_and_method_wrappers():
  """Builtins and method-wrappers do not loop forever and unwrap stably."""
  assert unwrap_callable(min) is min
  lst = []
  append_fn = lst.append
  assert unwrap_callable(append_fn) is append_fn
  add_fn = (1).__add__
  assert unwrap_callable(add_fn) is add_fn
  assert unwrap_callable(object.__init__) is object.__init__

  import dataclasses

  @dataclasses.dataclass
  class UnhashableCallable:
    items: list[int]

    def __call__(self, x: int) -> int:
      return x + sum(self.items)

  unhashable_obj = UnhashableCallable(items=[1, 2, 3])
  assert unwrap_callable(unhashable_obj) == UnhashableCallable.__call__


def test_get_type_hints_cached():
  """Cached type hints resolves annotations on callables."""

  def typed_fn(a: int, b: str) -> bool:
    return True

  hints = get_type_hints_cached(typed_fn)
  assert hints["a"] is int
  assert hints["b"] is str
  assert hints["return"] is bool


def test_is_context_type():
  """Detects Context and ToolContext aliases and Optional variations."""
  assert is_context_type(Context)
  assert is_context_type(ToolContext)
  assert is_context_type(Optional[Context])
  assert is_context_type(Optional[ToolContext])
  assert not is_context_type(int)
  assert not is_context_type(str)


def test_find_context_parameter():
  """Finds the first parameter annotated with Context or ToolContext."""

  def with_ctx(ctx: Context, query: str):
    pass

  def with_tool_ctx(tool_context: ToolContext, query: str):
    pass

  def without_ctx(query: str, limit: int = 10):
    pass

  assert find_context_parameter(with_ctx) == "ctx"
  assert find_context_parameter(with_tool_ctx) == "tool_context"
  assert find_context_parameter(without_ctx) is None


def test_callable_spec_attributes():
  """CallableSpec inspects doc, context parameter, and signature."""

  def my_tool(ctx: Context, count: int) -> str:
    """Does something useful."""
    return str(count)

  spec = CallableSpec(my_tool)
  assert spec.doc == "Does something useful."
  assert spec.context_param_name == "ctx"
  assert spec.signature is not None


def test_callable_spec_undocumented_function_doc_is_empty():
  """Undocumented functions have empty docstring rather than method wrapper doc."""
  from google.adk.tools.function_tool import FunctionTool

  def undocumented_fn(x: int) -> int:
    return x

  spec = CallableSpec(undocumented_fn)
  assert spec.doc == ""
  assert FunctionTool(undocumented_fn).description == ""


def test_callable_spec_callable_class_doc():
  """Callable objects resolve doc from __call__ or class appropriately."""

  class DocCallable:

    def __call__(self, x: int) -> int:
      """Call method docstring."""
      return x

  class UndocCallable:

    def __call__(self, x: int) -> int:
      return x

  assert CallableSpec(DocCallable()).doc == "Call method docstring."
  assert CallableSpec(UndocCallable()).doc == ""


def test_callable_spec_raises_on_unintrospectable_callable():
  """CallableSpec raises ValueError when signature of unintrospectable callable is accessed."""
  spec = CallableSpec(dir)
  with pytest.raises(ValueError):
    _ = spec.signature


def test_callable_spec_builtin_without_signature():
  """CallableSpec constructs successfully for builtins like min."""
  spec = CallableSpec(min)
  assert spec.doc != ""
  assert spec.context_param_name is None
  with pytest.raises(ValueError):
    _ = spec.signature


async def test_function_tool_supports_builtin_without_signature():
  """FunctionTool constructs and prepares arguments without error for builtins like min."""
  from google.adk.tools.function_tool import FunctionTool

  tool = FunctionTool(min)
  assert tool.name == "min"
  assert tool.description != ""
  assert tool._preprocess_args({"args": [1, 2]}) == {"args": [1, 2]}
  assert tool._prepare_invocation_args({"args": [1, 2]}, None) == {
      "args": [1, 2]
  }
  assert tool._get_mandatory_args() == []

  # Signature-less invocation failure returns structured error instead of crashing
  res = await tool.run_async(args={}, tool_context=None)
  assert "error" in res
  assert "Invoking `min()` failed" in res["error"]


async def test_function_tool_signatureless_reraises_internal_type_error():
  """FunctionTool re-raises internal TypeError inside a signature-less callable's body."""
  from google.adk.tools.function_tool import FunctionTool

  class FaultySignatureless:

    @property
    def __signature__(self):
      raise ValueError("no signature")

    def __call__(self, *args, **kwargs):
      return None + 1

  tool = FunctionTool(FaultySignatureless())
  assert not tool._spec.has_signature
  with pytest.raises(TypeError) as excinfo:
    await tool.run_async(args={}, tool_context=None)
  assert "unsupported operand type" in str(excinfo.value)


def test_callable_spec_raises_on_non_callable():
  """CallableSpec raises TypeError when passed a non-callable object."""
  with pytest.raises(TypeError):
    CallableSpec("not_a_callable")


def test_callable_spec_none():
  """CallableSpec constructs safely when passed None."""
  from google.adk.tools.function_tool import FunctionTool

  spec = CallableSpec(None)
  assert spec.func is None
  assert spec.unwrapped_func is None
  assert spec.doc == ""
  assert spec.context_param_name is None
  assert not spec.has_signature
  assert spec.type_hints == {}
  with pytest.raises(ValueError):
    _ = spec.signature

  tool = FunctionTool(None)
  assert tool.func is None
  assert tool.description == ""
  assert tool._get_declaration() is None


def test_type_hints_cache_and_unwrapping():
  """Verifies that type hints are cached and robustly unwrapped across partials and callables."""
  from google.adk.utils._callable_utils import _get_type_hints_for_unwrapped

  def my_func(x: int, y: str) -> bool:
    return True

  _get_type_hints_for_unwrapped.cache_clear()

  hints1 = get_type_hints_cached(my_func)
  assert hints1 == {"x": int, "y": str, "return": bool}

  hints2 = get_type_hints_cached(my_func)
  assert hints2 == {"x": int, "y": str, "return": bool}
  assert _get_type_hints_for_unwrapped.cache_info().hits == 1

  partial_func = functools.partial(my_func, x=1)
  hints3 = get_type_hints_cached(partial_func)
  assert hints3 == {"x": int, "y": str, "return": bool}
  assert _get_type_hints_for_unwrapped.cache_info().hits == 2

  class MyCallable:

    def __call__(self, z: float) -> None:
      pass

  obj = MyCallable()
  hints4 = get_type_hints_cached(obj)
  assert hints4 == {"z": float, "return": type(None)}


def test_callable_spec_forward_ref_recovery():
  """Forward reference recovers when defined in globals after spec construction."""

  # Function with forward reference to 'DeferredModel'
  def forward_ref_fn(item: "DeferredModel") -> int:
    return 1

  # At construction, DeferredModel is not in globals
  spec = CallableSpec(forward_ref_fn)
  assert spec.type_hints == {}

  # Define DeferredModel in globals
  class DeferredModel(pydantic.BaseModel):
    score: int

  globals()["DeferredModel"] = DeferredModel
  try:
    assert spec.type_hints == {"item": DeferredModel, "return": int}
  finally:
    globals().pop("DeferredModel", None)
