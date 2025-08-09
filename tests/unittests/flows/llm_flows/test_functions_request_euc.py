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

from typing import Any
from typing import Optional

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.llm_agent import Agent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.flows.llm_flows import functions
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ... import testing_utils


def function_call(function_call_id, name, args: dict[str, Any]) -> types.Part:
  part = types.Part.from_function_call(name=name, args=args)
  part.function_call.id = function_call_id
  return part


def test_function_request_euc():
  responses = [
      [
          types.Part.from_function_call(name='call_external_api1', args={}),
          types.Part.from_function_call(name='call_external_api2', args={}),
      ],
      [
          types.Part.from_text(text='response1'),
      ],
  ]

  auth_config1 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
          ),
      ),
  )
  auth_config2 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
          ),
      ),
  )

  mock_model = testing_utils.MockModel.create(responses=responses)

  def call_external_api1(tool_context: ToolContext) -> Optional[int]:
    tool_context.request_credential(auth_config1)

  def call_external_api2(tool_context: ToolContext) -> Optional[int]:
    tool_context.request_credential(auth_config2)

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[call_external_api1, call_external_api2],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test')
  assert events[0].content.parts[0].function_call is not None
  assert events[0].content.parts[1].function_call is not None
  auth_configs = list(events[2].actions.requested_auth_configs.values())
  exchanged_auth_config1 = auth_configs[0]
  exchanged_auth_config2 = auth_configs[1]
  assert exchanged_auth_config1.auth_scheme == auth_config1.auth_scheme
  assert (
      exchanged_auth_config1.raw_auth_credential
      == auth_config1.raw_auth_credential
  )
  assert (
      exchanged_auth_config1.exchanged_auth_credential.oauth2.auth_uri
      is not None
  )
  assert exchanged_auth_config2.auth_scheme == auth_config2.auth_scheme
  assert (
      exchanged_auth_config2.raw_auth_credential
      == auth_config2.raw_auth_credential
  )
  assert (
      exchanged_auth_config2.exchanged_auth_credential.oauth2.auth_uri
      is not None
  )
  function_call_ids = list(events[2].actions.requested_auth_configs.keys())

  for idx, part in enumerate(events[1].content.parts):
    request_euc_function_call = part.function_call
    assert request_euc_function_call is not None
    assert (
        request_euc_function_call.name
        == functions.REQUEST_EUC_FUNCTION_CALL_NAME
    )
    args = AuthToolArguments.model_validate(request_euc_function_call.args)

    assert args.function_call_id == function_call_ids[idx]
    args.auth_config.auth_scheme.model_extra.clear()
    assert args.auth_config.auth_scheme == auth_configs[idx].auth_scheme
    assert (
        args.auth_config.raw_auth_credential
        == auth_configs[idx].raw_auth_credential
    )


def test_function_get_auth_response():
  id_1 = 'id_1'
  id_2 = 'id_2'
  responses = [
      [
          function_call(id_1, 'call_external_api1', {}),
          function_call(id_2, 'call_external_api2', {}),
      ],
      [
          types.Part.from_text(text='response1'),
      ],
      [
          types.Part.from_text(text='response2'),
      ],
  ]

  mock_model = testing_utils.MockModel.create(responses=responses)
  function_invoked = 0

  auth_config1 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
          ),
      ),
  )
  auth_config2 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
          ),
      ),
  )

  auth_response1 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
          ),
      ),
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
              access_token='token1',
          ),
      ),
  )
  auth_response2 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
          ),
      ),
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
              access_token='token2',
          ),
      ),
  )

  def call_external_api1(tool_context: ToolContext) -> int:
    nonlocal function_invoked
    function_invoked += 1
    auth_response = tool_context.get_auth_response(auth_config1)
    if not auth_response:
      tool_context.request_credential(auth_config1)
      return
    assert auth_response == auth_response1.exchanged_auth_credential
    return 1

  def call_external_api2(tool_context: ToolContext) -> int:
    nonlocal function_invoked
    function_invoked += 1
    auth_response = tool_context.get_auth_response(auth_config2)
    if not auth_response:
      tool_context.request_credential(auth_config2)
      return
    assert auth_response == auth_response2.exchanged_auth_credential
    return 2

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[call_external_api1, call_external_api2],
  )
  runner = testing_utils.InMemoryRunner(agent)
  runner.run('test')
  request_euc_function_call_event = runner.session.events[-3]
  function_response1 = types.FunctionResponse(
      name=request_euc_function_call_event.content.parts[0].function_call.name,
      response=auth_response1.model_dump(),
  )
  function_response1.id = request_euc_function_call_event.content.parts[
      0
  ].function_call.id

  function_response2 = types.FunctionResponse(
      name=request_euc_function_call_event.content.parts[1].function_call.name,
      response=auth_response2.model_dump(),
  )
  function_response2.id = request_euc_function_call_event.content.parts[
      1
  ].function_call.id
  runner.run(
      new_message=types.Content(
          role='user',
          parts=[
              types.Part(function_response=function_response1),
              types.Part(function_response=function_response2),
          ],
      ),
  )

  assert function_invoked == 4
  request = mock_model.requests[-1]
  content = request.contents[-1]
  parts = content.parts
  assert len(parts) == 2
  assert parts[0].function_response.name == 'call_external_api1'
  assert parts[0].function_response.response == {'result': 1}
  assert parts[1].function_response.name == 'call_external_api2'
  assert parts[1].function_response.response == {'result': 2}


def test_function_get_auth_response_partial():
  id_1 = 'id_1'
  id_2 = 'id_2'
  responses = [
      [
          function_call(id_1, 'call_external_api1', {}),
          function_call(id_2, 'call_external_api2', {}),
      ],
      [
          types.Part.from_text(text='response1'),
      ],
      [
          types.Part.from_text(text='response2'),
      ],
      [
          types.Part.from_text(text='final response'),
      ],
  ]

  mock_model = testing_utils.MockModel.create(responses=responses)
  function_invoked = 0

  auth_config1 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
          ),
      ),
  )
  auth_config2 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
          ),
      ),
  )

  auth_response1 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
          ),
      ),
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_1',
              client_secret='oauth_client_secret1',
              access_token='token1',
          ),
      ),
  )
  auth_response2 = AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See, edit, share, and permanently delete all the'
                          ' calendars you can access using Google Calendar'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
          ),
      ),
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='oauth_client_id_2',
              client_secret='oauth_client_secret2',
              access_token='token2',
          ),
      ),
  )

  def call_external_api1(tool_context: ToolContext) -> int:
    nonlocal function_invoked
    function_invoked += 1
    auth_response = tool_context.get_auth_response(auth_config1)
    if not auth_response:
      tool_context.request_credential(auth_config1)
      return
    assert auth_response == auth_response1.exchanged_auth_credential
    return 1

  def call_external_api2(tool_context: ToolContext) -> int:
    nonlocal function_invoked
    function_invoked += 1
    auth_response = tool_context.get_auth_response(auth_config2)
    if not auth_response:
      tool_context.request_credential(auth_config2)
      return
    assert auth_response == auth_response2.exchanged_auth_credential
    return 2

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[call_external_api1, call_external_api2],
  )
  runner = testing_utils.InMemoryRunner(agent)
  runner.run('test')
  request_euc_function_call_event = runner.session.events[-3]
  function_response1 = types.FunctionResponse(
      name=request_euc_function_call_event.content.parts[0].function_call.name,
      response=auth_response1.model_dump(),
  )
  function_response1.id = request_euc_function_call_event.content.parts[
      0
  ].function_call.id

  function_response2 = types.FunctionResponse(
      name=request_euc_function_call_event.content.parts[1].function_call.name,
      response=auth_response2.model_dump(),
  )
  function_response2.id = request_euc_function_call_event.content.parts[
      1
  ].function_call.id
  runner.run(
      new_message=types.Content(
          role='user',
          parts=[
              types.Part(function_response=function_response1),
          ],
      ),
  )

  assert function_invoked == 3
  assert len(mock_model.requests) == 3
  request = mock_model.requests[-1]
  content = request.contents[-1]
  parts = content.parts
  assert len(parts) == 2
  assert parts[0].function_response.name == 'call_external_api1'
  assert parts[0].function_response.response == {'result': 1}
  assert parts[1].function_response.name == 'call_external_api2'
  assert parts[1].function_response.response == {'result': None}

  runner.run(
      new_message=types.Content(
          role='user',
          parts=[
              types.Part(function_response=function_response2),
          ],
      ),
  )
  assert function_invoked == 4
  assert len(mock_model.requests) == 4
  request = mock_model.requests[-1]
  content = request.contents[-1]
  parts = content.parts
  assert len(parts) == 2
  assert parts[0].function_response.name == 'call_external_api1'
  assert parts[0].function_response.response == {'result': 1}
  assert parts[1].function_response.name == 'call_external_api2'
  assert parts[1].function_response.response == {'result': 2}
