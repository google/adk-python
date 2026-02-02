# Copyright 2025 Google LLC
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

import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from mcp.server.fastmcp import Context
from mcp.server.fastmcp import FastMCP
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected OAuth token for testing
VALID_TOKEN = 'test_access_token_12345'

# Create FastMCP server
mcp = FastMCP('OAuth Protected MCP Server', host='localhost', port=3001)


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


# Create custom FastAPI app to add auth middleware for list_tools
app = FastAPI()


@app.middleware('http')
async def auth_middleware(request: Request, call_next):
  """Middleware to validate auth on all MCP endpoints."""
  # Check if this is an MCP request
  if request.url.path.startswith('/mcp'):
    if not validate_auth_header(request):
      raise HTTPException(status_code=401, detail='Unauthorized')
  return await call_next(request)


if __name__ == '__main__':
  print(f'Starting OAuth Protected MCP server on http://localhost:3001')
  print(f'Expected token: Bearer {VALID_TOKEN}')
  print(
      'This server requires authentication for both tool listing and calling.'
  )

  # Run with streamable-http transport
  mcp.run(transport='streamable-http')
