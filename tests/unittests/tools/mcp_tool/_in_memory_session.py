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

"""An in-memory client session for testing `to_mcp_server`.

`mcp.shared.memory.create_connected_server_and_client_session` does the same
thing, but it is a convenience wrapper the SDK does not promise. This builds the
session from the four pieces underneath it, each of which the SDK does promise:
the memory stream pair, the low-level server's `run`, its initialization
options, and `ClientSession`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from typing import AsyncGenerator

import anyio
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams


def _lowlevel_server(server: Any) -> Any:
  """Returns the low-level server that `server` wraps, or `server` itself."""
  # The high-level server holds the low-level one privately and offers no
  # accessor. The SDK's own in-memory helper reaches for it the same way.
  for name in ("_mcp_server", "_lowlevel_server"):
    wrapped = getattr(server, name, None)
    if wrapped is not None:
      return wrapped
  return server


@asynccontextmanager
async def connected_client_session(
    server: Any,
    *,
    raise_exceptions: bool = True,
) -> AsyncGenerator[ClientSession, None]:
  """Yields an initialized `ClientSession` talking to `server` in memory.

  Args:
    server: The server to connect to, high-level or low-level.
    raise_exceptions: Whether a handler that raises should take the server down
      rather than return the error to the client. On by default, so a test sees
      the traceback instead of an error result.

  Yields:
    A `ClientSession` that has completed the initialize handshake.
  """
  server = _lowlevel_server(server)

  async with create_client_server_memory_streams() as (
      client_streams,
      server_streams,
  ):
    client_read, client_write = client_streams
    server_read, server_write = server_streams

    async with anyio.create_task_group() as task_group:
      task_group.start_soon(
          lambda: server.run(
              server_read,
              server_write,
              server.create_initialization_options(),
              raise_exceptions=raise_exceptions,
          )
      )
      try:
        async with ClientSession(
            read_stream=client_read,
            write_stream=client_write,
        ) as session:
          await session.initialize()
          yield session
      finally:
        task_group.cancel_scope.cancel()
