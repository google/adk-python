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

"""The HTTP client library ADK's MCP transport builds on, open-source build.

The MCP SDK does not merely accept an HTTP client, it types one: the client ADK
constructs is handed straight to the SDK's transport. So the flavor here is not
a free choice -- it has to be the one the installed SDK was built against. MCP
1.x pairs with `httpx`; 2.x pairs with `httpx2`, the continuation of the same
library under Pydantic's stewardship.

The pin admits both majors, so the choice is made here, at import. It keys off
which SDK is installed rather than which HTTP library is importable: an MCP 1.x
install can still have `httpx2` present for an unrelated reason, and choosing
it there would hand the SDK a client it cannot use.

Only `tools/mcp_tool` goes through here. The rest of ADK talks to `httpx`
directly and is unaffected by the SDK's choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._mcp import IS_MCP_SDK_V2

# A type checker always takes the `httpx` branch. Only one flavor is installed
# where it runs, and an import it cannot resolve becomes `Any` -- so following
# the other branch drops every annotation here and rejects the two transports in
# `mcp_session_manager` that subclass one. Both libraries present the same API to
# ADK, and `httpx` is the one the lockfile resolves. The cost is that nothing is
# ever checked against `httpx2`, so a divergence between the two is invisible to
# the checker; the tests catch that instead, running against a real 2.x install.
if TYPE_CHECKING or not IS_MCP_SDK_V2:
  from httpx import AsyncBaseTransport as AsyncBaseTransport
  from httpx import AsyncByteStream as AsyncByteStream
  from httpx import AsyncClient as AsyncClient
  from httpx import Auth as Auth
  from httpx import Headers as Headers
  from httpx import HTTPStatusError as HTTPStatusError
  from httpx import Request as Request
  from httpx import Response as Response
  from httpx import Timeout as Timeout
  from httpx import URL as URL
else:
  from httpx2 import AsyncBaseTransport as AsyncBaseTransport
  from httpx2 import AsyncByteStream as AsyncByteStream
  from httpx2 import AsyncClient as AsyncClient
  from httpx2 import Auth as Auth
  from httpx2 import Headers as Headers
  from httpx2 import HTTPStatusError as HTTPStatusError
  from httpx2 import Request as Request
  from httpx2 import Response as Response
  from httpx2 import Timeout as Timeout
  from httpx2 import URL as URL

__all__ = [
    "URL",
    "AsyncBaseTransport",
    "AsyncByteStream",
    "AsyncClient",
    "Auth",
    "HTTPStatusError",
    "Headers",
    "Request",
    "Response",
    "Timeout",
]
