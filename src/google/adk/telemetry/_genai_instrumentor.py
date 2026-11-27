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

from ..utils.model_name_utils import is_gemini_model
from .context import _read_experimental_genai_semconv
from .context import TelemetryConfig

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])

# Where the instrumentor's wrapper functions live, in every release.
_MODULE_PATH = "opentelemetry/instrumentation/google_genai"

# The newest opentelemetry-util-genai release verified not to record the
# gen_ai.client.inference.* token metrics. The instrumentor >= 1.0 records its
# metrics through it, and < 1.0 (which pins util-genai < 0.4) records none of
# these, so up to here ADK records them alongside it without double counting.
_LAST_UTIL_GENAI_WITHOUT_INFERENCE_TOKEN_METRICS = Version("1.3b0")


def _is_instrumentor_code(function: object) -> bool:
  code = getattr(function, "__code__", None)
  if code is None:
    return False
  return _MODULE_PATH in code.co_filename.replace("\\", "/")


def _is_instrumentor_layer(function: object) -> bool:
  """Whether ``function`` is a layer the instrumentor put over the SDK.

  How each release wraps the SDK's generate content method:

  - < 1.0:  instrumentor function --wraps--> SDK function
            The SDK function is replaced outright, so the layer's code is the
            instrumentor's own.
  - >= 1.0: wrapt proxy --wrapper--> instrumentor function
                        --wraps----> SDK function
            The proxy reports the SDK function's code, which >= 1.0b1 marks
            with the instrumentor's file name.

  That mark alone misses 1.0b0, and it mistakes any wrapt proxy above the
  instrumentor for the instrumentor, since that proxy also reports the marked
  code. So read the proxy's wrapper, and fall back to the mark only if wrapt
  stops exposing it.
  """
  wrapper = getattr(function, "_self_wrapper", None)
  return _is_instrumentor_code(function if wrapper is None else wrapper)


def is_instrumented() -> bool:
  """Whether the instrumentor wraps ``Models.generate_content``."""
  maybe_wrapped_function = Models.generate_content
  while wrapped := getattr(maybe_wrapped_function, "__wrapped__", None):
    if _is_instrumentor_layer(maybe_wrapped_function):
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


def should_emit_inference_token_metrics(model: str | None) -> bool:
  """Whether ADK records the inference token metrics for a call to `model`."""
  # The instrumentor only covers Gemini calls. Without it, ADK is the only one
  # recording these metrics.
  if not (is_instrumented() and is_gemini_model(model)):
    return True
  try:
    installed = Version(metadata.version("opentelemetry-util-genai"))
  except metadata.PackageNotFoundError:
    return True
  return installed <= _LAST_UTIL_GENAI_WITHOUT_INFERENCE_TOKEN_METRICS


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
  return cast(_CallableT, _without_instrumentor(method))


def _without_instrumentor(
    method: Callable[..., object],
) -> Callable[..., object]:
  """``method`` with the instrumentor's layer left out, or as is if it can't be.

  Layers above it are kept, as long as they are wrapt proxies: their wrapper is
  handed what lies beneath the instrumentor instead. Beneath any other kind,
  util-genai >= 1.3b0 still stands it down, because ADK's inference span marks
  the context; older ones record as well.
  """
  inner = getattr(method, "__wrapped__", None)
  if inner is None:
    return method
  if _is_instrumentor_layer(method):
    # An exact type check, since wrapt proxies report the wrapped class.
    if type(method) is types.MethodType:
      # < 1.0: a plain replacement, which wraps the unbound SDK function, or
      # an unbound wrapt proxy beneath it.
      return types.MethodType(inner, method.__self__)
    # >= 1.0: a bound wrapt proxy, which wraps the bound SDK method.
    return cast(Callable[..., object], inner)
  wrapper = getattr(method, "_self_wrapper", None)
  beneath = _without_instrumentor(inner)  # pyright: ignore[reportAny]
  if wrapper is None or beneath is inner:
    return method
  instance = getattr(method, "_self_instance", None)
  return lambda *args, **kwargs: wrapper(beneath, instance, args, kwargs)
