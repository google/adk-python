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

"""Tests for auth scheme helpers."""

from __future__ import annotations

from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlowImplicit
from fastapi.openapi.models import OAuthFlowPassword
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_schemes import OAuthGrantType
import pytest

_TOKEN_URL = 'https://example.com/token'
_AUTH_URL = 'https://example.com/authorize'


@pytest.mark.parametrize(
    ('flows', 'expected'),
    [
        pytest.param(
            OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl=_TOKEN_URL, scopes={}
                )
            ),
            OAuthGrantType.CLIENT_CREDENTIALS,
            id='client-credentials',
        ),
        pytest.param(
            OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl=_AUTH_URL, tokenUrl=_TOKEN_URL, scopes={}
                )
            ),
            OAuthGrantType.AUTHORIZATION_CODE,
            id='authorization-code',
        ),
        pytest.param(
            OAuthFlows(
                implicit=OAuthFlowImplicit(
                    authorizationUrl=_AUTH_URL, scopes={}
                )
            ),
            OAuthGrantType.IMPLICIT,
            id='implicit',
        ),
        pytest.param(
            OAuthFlows(
                password=OAuthFlowPassword(tokenUrl=_TOKEN_URL, scopes={})
            ),
            OAuthGrantType.PASSWORD,
            id='password',
        ),
    ],
)
def test_from_flow_maps_each_configured_flow_to_its_grant_type(flows, expected):
  assert OAuthGrantType.from_flow(flows) == expected


def test_from_flow_without_any_configured_flow_returns_none():
  """An OAuth2 scheme declaring no flow has no grant type to exchange with."""
  assert OAuthGrantType.from_flow(OAuthFlows()) is None


def test_grant_type_values_are_the_oauth2_wire_names():
  # These strings go on the wire as the OAuth2 `grant_type` parameter, so
  # they must stay exactly as the spec names them.
  assert OAuthGrantType.CLIENT_CREDENTIALS.value == 'client_credentials'
  assert OAuthGrantType.AUTHORIZATION_CODE.value == 'authorization_code'
  assert OAuthGrantType.IMPLICIT.value == 'implicit'
  assert OAuthGrantType.PASSWORD.value == 'password'
