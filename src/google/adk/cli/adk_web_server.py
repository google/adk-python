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

import logging

from typing_extensions import deprecated

from .api_server import _parse_cors_origins
from .api_server import RunAgentRequest
from .dev_server import DevServer

logger = logging.getLogger("google_adk." + __name__)


@deprecated(
    "AdkWebServer is deprecated and has been refactored into ApiServer and"
    " DevServer. Use DevServer instead."
)
class AdkWebServer(DevServer):
  """Deprecated wrapper class around DevServer for backward compatibility."""

  pass
