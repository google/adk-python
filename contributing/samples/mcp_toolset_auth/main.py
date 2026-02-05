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

"""Test script for MCP Toolset OAuth Authentication Flow.

This script demonstrates the two-phase tool discovery flow:
1. First invocation: Agent tries to get tools, auth is required, returns auth
   request event (adk_request_credential)
2. User provides OAuth credentials (simulated)
3. Second invocation: Agent has credentials, can list and call tools

Usage:
  # Start the MCP server first (in another terminal):
  PYTHONPATH=src python contributing/samples/mcp_toolset_auth/oauth_mcp_server.py

  # Run the demo:
  PYTHONPATH=src python contributing/samples/mcp_toolset_auth/main.py
"""

from __future__ import annotations

import asyncio

from agent import auth_credential
from agent import auth_scheme
from agent import mcp_toolset
from agent import root_agent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


async def run_demo():
  """Run demo with real MCP server."""
  print('=' * 60)
  print('MCP Toolset OAuth Authentication Demo')
  print('=' * 60)
  print('\nNote: Make sure the MCP server is running:')
  print('  python oauth_mcp_server.py\n')

  # Create session service and runner
  session_service = InMemorySessionService()
  runner = Runner(
      agent=root_agent,
      app_name='toolset_auth_demo',
      session_service=session_service,
  )

  # Create a session
  session = await session_service.create_session(
      app_name='toolset_auth_demo',
      user_id='test_user',
  )

  print(f'Session created: {session.id}')
  print('\n--- Phase 1: Initial request (no credentials) ---\n')

  # First invocation - should trigger auth request
  user_message = 'List all users'
  print(f'User: {user_message}')

  events = []
  auth_function_call_id = None
  max_events = 10

  try:
    async for event in runner.run_async(
        session_id=session.id,
        user_id='test_user',
        new_message=types.Content(
            role='user',
            parts=[types.Part(text=user_message)],
        ),
    ):
      events.append(event)
      print(f'\nEvent from {event.author}:')
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            print(f'  Text: {part.text}')
          if part.function_call:
            print(f'  Function call: {part.function_call.name}')
            if part.function_call.name == 'adk_request_credential':
              auth_function_call_id = part.function_call.id

      if len(events) >= max_events:
        print(f'\n** SAFETY LIMIT ({max_events} events) **')
        break

  except Exception as e:
    print(f'\nError: {e}')
    print('Make sure the MCP server is running!')
    await mcp_toolset.close()
    return

  if auth_function_call_id:
    print('\n** Auth request detected! **')
    print('\n--- Phase 2: Provide OAuth credentials ---\n')

    # Simulate user providing OAuth credentials after completing OAuth flow
    auth_response = AuthConfig(
        auth_scheme=auth_scheme,
        raw_auth_credential=auth_credential,
        exchanged_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                access_token='test_access_token_12345',
            ),
        ),
    )

    print('Providing access token: test_access_token_12345')

    auth_response_message = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name='adk_request_credential',
                    id=auth_function_call_id,
                    response=auth_response.model_dump(exclude_none=True),
                )
            )
        ],
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id='test_user',
        new_message=auth_response_message,
    ):
      print(f'\nEvent from {event.author}:')
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            text = (
                part.text[:200] + '...' if len(part.text) > 200 else part.text
            )
            print(f'  Text: {text}')
          if part.function_call:
            print(f'  Function call: {part.function_call.name}')
  else:
    print('\n** No auth request - credentials may already be available **')

  print('\n' + '=' * 60)
  print('Demo completed')
  print('=' * 60)

  await mcp_toolset.close()


if __name__ == '__main__':
  asyncio.run(run_demo())
