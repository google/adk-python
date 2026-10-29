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
import sys
from typing import Optional

from google.adk.agents.context import Context
from google.adk.tools.tool_context import ToolContext
from google.adk.utils._callable_utils import CallableSpec
from google.adk.utils._callable_utils import get_best_effort_type_hints
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


def _wrapped_with_doc(a: int, b: int) -> int:
  """Wrapped function docstring."""
  return a + b


class _DocCallable:

  def __call__(self, a: int, b: int) -> int:
    """Call method docstring."""
    return a + b


_documented_partial = functools.partial(_wrapped_with_doc, 1)
_documented_partial.__doc__ = "Partial docstring."


@pytest.mark.parametrize(
    ("partial_fn", "expected_doc"),
    [
        (
            functools.partial(_wrapped_with_doc, 1),
            "Wrapped function docstring.",
        ),
        (
            functools.partial(_DocCallable(), 1),
            "Call method docstring.",
        ),
        (
            _documented_partial,
            "Partial docstring.",
        ),
    ],
)
def test_callable_spec_partial_doc(
    partial_fn: functools.partial[int], expected_doc: str
) -> None:
  """A partial resolves doc from the callable it wraps unless it has its own."""
  assert CallableSpec(partial_fn).doc == expected_doc


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


def test_callable_spec_is_generator_detects_sync_async_and_wrapped_generators():
  """CallableSpec.is_generator inspects the invoked callable rather than following __wrapped__."""

  def sync_gen(x: int = 1):
    yield x

  async def async_gen():
    yield 1

  @functools.wraps(sync_gen)
  def list_wrapper(*args, **kwargs):
    return list(sync_gen(*args, **kwargs))

  def regular_fn():
    return 1

  assert CallableSpec(sync_gen).is_generator is True
  assert CallableSpec(async_gen).is_generator is True
  assert CallableSpec(functools.partial(sync_gen, 2)).is_generator is True
  assert CallableSpec(list_wrapper).is_generator is False
  assert CallableSpec(regular_fn).is_generator is False
  assert CallableSpec(None).is_generator is False


def test_callable_spec_resolves_type_checking_context():
  """Resolves Context under TYPE_CHECKING when absent from runtime globals."""
  import datetime
  import types

  # Simulate a module with `from __future__ import annotations` where `Context`
  # was imported under `if TYPE_CHECKING:` (absent from runtime module globals),
  # while `datetime` is present in module globals.
  isolated_globals = {"__builtins__": __builtins__, "datetime": datetime}

  fn_only_ctx = types.FunctionType(
      (lambda ctx, start_date: str(start_date)).__code__,
      isolated_globals,
      name="fn_only_ctx",
  )
  fn_only_ctx.__annotations__ = {
      "ctx": "Context",
      "start_date": "str | datetime.date",
      "return": "str",
  }

  spec = CallableSpec(fn_only_ctx)
  assert spec.context_param_name == "ctx"
  assert spec.type_hints["ctx"] is Context
  assert spec.type_hints["start_date"] == (str | datetime.date)


def test_callable_spec_best_effort_type_hints_preserves_resolvable_siblings():
  """Preserves sibling hints via get_type_hints semantics when a parameter raises NameError."""
  import datetime
  import types

  from typing_extensions import Annotated

  isolated_globals = {
      "__builtins__": __builtins__,
      "Annotated": Annotated,
      "Optional": Optional,
      "datetime": datetime,
  }

  fn_with_unresolved = types.FunctionType(
      (
          lambda ctx, start_date, extra=None, helper=None: str(start_date)
      ).__code__,
      isolated_globals,
      name="fn_with_unresolved",
      argdefs=(None, None),
  )
  fn_with_unresolved.__annotations__ = {
      "ctx": "Annotated[Context, 'meta']",
      "start_date": "datetime.date",
      "extra": Optional["ToolContext"],
      "helper": "UnresolvedHelper | None",
      "return": "None",
  }

  spec = CallableSpec(fn_with_unresolved)
  assert spec.context_param_name == "ctx"
  best_effort = spec.get_best_effort_type_hints()
  assert best_effort["ctx"] is Context
  assert best_effort["start_date"] is datetime.date
  assert best_effort["extra"] == Optional[ToolContext]
  assert best_effort["return"] is type(None)
  assert "helper" not in best_effort


def test_function_tool_validates_resolvable_sibling_with_unresolved_annotation():
  """FunctionTool coerces resolvable parameters when a sibling parameter is unresolvable."""
  import datetime
  import types

  from google.adk.features import FeatureName
  from google.adk.features._feature_registry import temporary_feature_override
  from google.adk.tools.function_tool import FunctionTool

  isolated_globals = {"__builtins__": __builtins__, "datetime": datetime}
  fn_with_unresolved = types.FunctionType(
      (lambda ctx, start_date, helper=None: str(start_date)).__code__,
      isolated_globals,
      name="fn_with_unresolved",
      argdefs=(None,),
  )
  fn_with_unresolved.__annotations__ = {
      "ctx": "Context",
      "start_date": "datetime.date",
      "helper": "UnresolvedHelper | None",
      "return": "str",
  }

  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    tool = FunctionTool(fn_with_unresolved)
    processed, errors = tool._preprocess_args_with_validation(
        {"start_date": "2026-10-19", "helper": "raw"}
    )
    assert not errors
    assert processed["start_date"] == datetime.date(2026, 10, 19)
    assert processed["helper"] == "raw"


def test_type_hints_retry_preserves_user_defined_context_class():
  """Module-level user Context class is not shadowed when retrying NameError."""
  import types

  from google.adk.tools.tool_context import ToolContext

  class UserContext:
    pass

  isolated_globals = {"__builtins__": __builtins__, "Context": UserContext}
  fn = types.FunctionType(
      (lambda user_ctx, adk_ctx: str(user_ctx)).__code__,
      isolated_globals,
      name="fn_user_context",
  )
  fn.__annotations__ = {
      "user_ctx": "Context",
      "adk_ctx": "ToolContext",
      "return": "str",
  }

  spec = CallableSpec(fn)
  assert spec.type_hints["user_ctx"] is UserContext
  assert spec.type_hints["adk_ctx"] is ToolContext
  assert spec.context_param_name == "adk_ctx"


@pytest.mark.parametrize(
    "bad_annotation",
    [
        "int | 1",
        "datetime.MissingAttribute",
    ],
)
def test_type_hints_retry_catches_type_and_attribute_errors(
    bad_annotation: str,
):
  """Retrying get_type_hints after NameError catches TypeError and AttributeError."""
  import datetime
  import types

  isolated_globals = {"__builtins__": __builtins__, "datetime": datetime}
  fn = types.FunctionType(
      (lambda ctx, start_date, bad=None: str(start_date)).__code__,
      isolated_globals,
      name="fn_retry_error",
      argdefs=(None,),
  )
  fn.__annotations__ = {
      "ctx": "Context",
      "start_date": "datetime.date",
      "bad": bad_annotation,
      "return": "str",
  }

  assert get_type_hints_cached(fn) == {}
  spec = CallableSpec(fn)
  assert spec.context_param_name == "ctx"
  assert spec.type_hints == {}
  assert spec.get_best_effort_type_hints() == {
      "ctx": Context,
      "start_date": datetime.date,
      "return": str,
  }


def test_callable_spec_memoizes_best_effort_type_hints():
  """CallableSpec memoizes fallback per-parameter type hint evaluation across calls."""
  import types

  lookups = 0

  class _CountingNamespace:

    @property
    def ResolvedType(self) -> type[int]:
      nonlocal lookups
      lookups += 1
      return int

  isolated_globals = {"__builtins__": __builtins__, "ns": _CountingNamespace()}
  fn = types.FunctionType(
      (lambda value, helper=None: value).__code__,
      isolated_globals,
      name="fn_memoized_best_effort",
      argdefs=(None,),
  )
  fn.__annotations__ = {
      "value": "ns.ResolvedType",
      "helper": "UnresolvedHelper",
  }

  spec = CallableSpec(fn)
  lookups = 0

  first = spec.get_best_effort_type_hints()
  lookups_after_first = lookups
  second = spec.get_best_effort_type_hints()
  third = spec.get_best_effort_type_hints()

  assert first == {"value": int}
  assert second == {"value": int}
  assert third == {"value": int}
  assert lookups_after_first > 0
  assert lookups == lookups_after_first


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="PEP 649 deferred evaluation of annotations requires Python 3.14+",
)
def test_best_effort_type_hints_handles_pep649_annotation_errors():
  """get_best_effort_type_hints catches NameError and TypeError on PEP 649 __annotations__."""
  from unittest.mock import MagicMock

  ns: dict[str, object] = {
      "__builtins__": __builtins__,
      "mock_type": MagicMock(spec=object),
  }
  exec(
      compile(
          """
def fn_name_error(x: UndefinedPep649Type = None):
  return x

class CallableNameError:
  def __call__(self, x: UndefinedPep649Type = None):
    return x

def fn_type_error(x: mock_type | None = None):
  return x

class CallableTypeError:
  def __call__(self, x: mock_type | None = None):
    return x
""",
          "<pep649_test>",
          "exec",
          dont_inherit=True,
      ),
      ns,
  )

  fn_name_error = ns["fn_name_error"]
  callable_name_error = ns["CallableNameError"]()  # type: ignore[operator]
  fn_type_error = ns["fn_type_error"]
  callable_type_error = ns["CallableTypeError"]()  # type: ignore[operator]

  assert get_best_effort_type_hints(fn_name_error) == {}  # type: ignore[arg-type]
  assert get_best_effort_type_hints(callable_name_error) == {}
  assert get_best_effort_type_hints(fn_type_error) == {}  # type: ignore[arg-type]
  assert get_best_effort_type_hints(callable_type_error) == {}
  assert CallableSpec(fn_type_error).get_best_effort_type_hints() == {}  # type: ignore[arg-type]
  assert CallableSpec(callable_type_error).get_best_effort_type_hints() == {}
