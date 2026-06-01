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

"""A2A agents package."""

from ...utils._dependency import missing_extra

__all__ = [
    "A2aRemoteAgentConfig",
    "ParametersConfig",
    "RequestInterceptor",
]


def __getattr__(name: str):
  if name in [
      "A2aRemoteAgentConfig",
      "ParametersConfig",
      "RequestInterceptor",
  ]:
    try:
      from .config import A2aRemoteAgentConfig
      from .config import ParametersConfig
      from .config import RequestInterceptor

      if name == "A2aRemoteAgentConfig":
        return A2aRemoteAgentConfig
      elif name == "ParametersConfig":
        return ParametersConfig
      elif name == "RequestInterceptor":
        return RequestInterceptor
    except ImportError as e:
      raise missing_extra("a2a-sdk", "a2a") from e
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
