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

import functools
import inspect
import logging
from typing import Callable
from typing import Concatenate
from typing import ParamSpec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .context import _ExperimentalFeature
  from .context import TelemetryConfig

logger = logging.getLogger("google_adk." + __name__)

_Args = ParamSpec("_Args")


def experimental_telemetry(
    gate: _ExperimentalFeature | list[_ExperimentalFeature],
) -> Callable[
    [Callable[_Args, None]], Callable[Concatenate[TelemetryConfig, _Args], None]
]:
  """Gates a function on one or more experimental telemetry features.

  The decorated function gains a leading positional-only ``TelemetryConfig``,
  and runs only when that config enables every feature named here.

  Args:
    gate: The experimental feature, or features, the function belongs to. Naming
      several means all must be enabled for the function to run.

  Returns:
    A decorator that prepends the config parameter and check to the function it
    wraps.
  """
  gates = [gate] if isinstance(gate, str) else gate

  def decorator(
      func: Callable[_Args, None],
  ) -> Callable[Concatenate[TelemetryConfig, _Args], None]:
    func_name = getattr(func, "__qualname__", None) or repr(func)

    @functools.wraps(func)
    def wrapper(
        config: TelemetryConfig,
        /,
        *args: _Args.args,
        **kwargs: _Args.kwargs,
    ) -> None:
      enabled: bool
      if gates:
        enabled = all(
            config._experimental_feature_enabled(feature) for feature in gates
        )
      else:
        enabled = config.should_emit_experimental_telemetry

      if enabled:
        func(*args, **kwargs)
      else:
        logger.debug("Skipping experimental telemetry function %s", func_name)

    # Update the signature to include the config parameter.
    signature = inspect.signature(func)
    wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(
                "config",
                inspect.Parameter.POSITIONAL_ONLY,
                annotation="TelemetryConfig",
            ),
            *signature.parameters.values(),
        ]
    )
    return wrapper

  return decorator
