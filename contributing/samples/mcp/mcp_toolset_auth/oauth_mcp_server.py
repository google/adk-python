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

"""MCP Server that requires OAuth Bearer token for both tool listing and calling.

This server validates the Authorization header on every request including:
- Tool listing (list_tools endpoint)
- Tool calling (call_tool endpoint)

This is used to test the toolset authentication feature in ADK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

# MCP 2.0 renamed this server class. ADK supports both majors, so this sample
# does too.
try:
  from mcp.server.mcpserver import Context
  from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
  from mcp.server.fastmcp import Context
  from mcp.server.fastmcp import FastMCP

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('google_adk.' + __name__)

# Expected OAuth token for testing
VALID_TOKEN = 'test_access_token_12345'

HOST = 'localhost'
PORT = 3001

# This sample serves through uvicorn at the bottom of the file rather than
# mcp.run(), so the bind address goes there. 2.0 moved it off the constructor.
mcp = FastMCP('OAuth Protected MCP Server')


def validate_auth_header(request: Request) -> bool:
  """Validate the Authorization header contains a valid Bearer token."""
  auth_header = request.headers.get('authorization', '')
  if not auth_header.startswith('Bearer '):
    logger.warning('Missing or invalid Authorization header: %s', auth_header)
    return False

  token = auth_header[7:]  # Remove 'Bearer ' prefix
  if token != VALID_TOKEN:
    logger.warning('Invalid token: %s', token)
    return False

  logger.info('Valid token received')
  return True


@mcp.tool(description='Get user profile information. Requires authentication.')
def get_user_profile(user_id: str, context: Context) -> dict:
  """Return user profile data for the given user ID."""
  logger.info('get_user_profile called for user: %s', user_id)

  if context.request_context and context.request_context.request:
    if not validate_auth_header(context.request_context.request):
      return {'error': 'Unauthorized - invalid or missing token'}

  # Mock user data
  users = {
      'user1': {'id': 'user1', 'name': 'Alice', 'email': 'alice@example.com'},
      'user2': {'id': 'user2', 'name': 'Bob', 'email': 'bob@example.com'},
  }

  if user_id in users:
    return users[user_id]
  return {'error': f'User {user_id} not found'}


@mcp.tool(description='List all available users. Requires authentication.')
def list_users(context: Context) -> dict:
  """Return a list of all users."""
  logger.info('list_users called')

  if context.request_context and context.request_context.request:
    if not validate_auth_header(context.request_context.request):
      return {'error': 'Unauthorized - invalid or missing token'}

  return {
      'users': [
          {'id': 'user1', 'name': 'Alice'},
          {'id': 'user2', 'name': 'Bob'},
      ]
  }


# FastMCP's own Starlette app is what serves the /mcp endpoint, so mounting it
# under a FastAPI app is what puts the auth middleware in front of every MCP
# request, tool listing included. A mounted app's lifespan is not run by the
# mount, so the session manager the endpoint depends on is started from the
# FastAPI lifespan instead.
mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
  async with mcp.session_manager.run():
    yield


# Create custom FastAPI app to add auth middleware for list_tools
app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def auth_middleware(request: Request, call_next):
  """Middleware to validate auth on all MCP endpoints."""
  # Check if this is an MCP request
  if request.url.path.startswith('/mcp'):
    if not validate_auth_header(request):
      # Returned rather than raised: an exception from HTTP middleware escapes
      # the exception handlers and becomes a 500.
      return JSONResponse(status_code=401, content={'detail': 'Unauthorized'})
  return await call_next(request)


app.mount('/', mcp_app)


if __name__ == '__main__':
  print(f'Starting OAuth Protected MCP server on http://{HOST}:{PORT}')
  print(f'Expected token: Bearer {VALID_TOKEN}')
  print(
      'This server requires authentication for both tool listing and calling.'
  )

  uvicorn.run(app, host=HOST, port=PORT)
