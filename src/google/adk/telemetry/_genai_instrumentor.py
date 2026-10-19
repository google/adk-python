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

"""Introspection of opentelemetry-instrumentation-google-genai's wrappers."""

from __future__ import annotations

import functools
from importlib import metadata
import types
from typing import Callable
from typing import cast
from typing import TypeVar

from google.genai.models import Models
from packaging.version import Version

from .context import _read_experimental_genai_semconv
from .context import TelemetryConfig

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])

# >= 1.0 rewrites the `co_filename` of each SDK function it wraps to this path
# (`_set_co_filename`, which it keeps for ADK); < 1.0 wrappers live there.
_MODULE_PATH = "opentelemetry/instrumentation/google_genai"


def _is_instrumentor_code(function: object) -> bool:
  code = getattr(function, "__code__", None)
  if code is None:
    return False
  return _MODULE_PATH in code.co_filename.replace("\\", "/")


def is_instrumented() -> bool:
  """Whether the instrumentor wraps ``Models.generate_content``."""
  maybe_wrapped_function = Models.generate_content
  while wrapped := getattr(maybe_wrapped_function, "__wrapped__", None):
    if _is_instrumentor_code(maybe_wrapped_function):
      return True
    maybe_wrapped_function = wrapped  # pyright: ignore[reportAny]
  return False


@functools.cache
def _has_stable_mode() -> bool:
  """Whether the installed instrumentor is < 1.0, which still has a stable mode."""
  try:
    version = metadata.version("opentelemetry-instrumentation-google-genai")
  except metadata.PackageNotFoundError:
    return False
  return Version(version).major < 1


def records_inference(telemetry_config: TelemetryConfig) -> bool:
  """Whether the instrumentor, not ADK, records a Gemini inference.

  Only when it records the semconv the request wants: the experimental one,
  which < 1.0 records only under ``OTEL_SEMCONV_STABILITY_OPT_IN``.
  """
  return (
      is_instrumented()
      and telemetry_config.should_use_experimental_genai_semconv
      and (not _has_stable_mode() or _read_experimental_genai_semconv())
  )


def maybe_without_genai_instrumentation(method: _CallableT) -> _CallableT:
  """Returns ``method`` minus the instrumentor, if ADK records the inference.

  Short-term: the instrumentor >= 1.0 records only semconv 1.44, so to keep
  supporting semconv 1.36 ADK records the inference itself and bypasses the
  instrumentor. Delete when telemetry schema v1 is dropped.

  Not for the Interactions API: ADK has no instrumentation of its own for it,
  so the instrumentor is the only thing recording those calls.
  """
  # Lazy: both import this module.
  from . import _instrumentation
  from . import tracing

  scope = _instrumentation.inference_scope()
  if scope is None or not isinstance(scope.span, tracing.GenerateContentSpan):
    return method
  # Peels the instrumentor only when it is the outermost layer over a
  # plain function. Any wrapt wrapper stacked with it hides which layer is its,
  # and both record.
  inner = getattr(method, "__wrapped__", None)
  # `type() is`, not `isinstance`: wrapt proxies `__class__`.
  if type(inner) is types.MethodType and _is_instrumentor_code(inner.__func__):
    # >= 1.0: a wrapt wrapper, which stamps the method it wraps.
    return cast(_CallableT, inner)
  if (
      isinstance(method, types.MethodType)
      and type(inner) is types.FunctionType
      and _is_instrumentor_code(method.__func__)
  ):
    # < 1.0: a functools.wraps replacement, whose `__wrapped__` is unbound.
    return cast(_CallableT, types.MethodType(inner, method.__self__))
  return method
