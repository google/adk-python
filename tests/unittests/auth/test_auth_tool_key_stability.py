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

import unittest
from unittest.mock import Mock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig


class TestAuthToolKeyStability(unittest.TestCase):

  def test_key_stability_with_different_secrets(self):
    from google.adk.auth.auth_schemes import AuthSchemeType
    from google.adk.auth.auth_schemes import OAuth2

    # Consistent scheme for both
    auth_scheme = OAuth2(type=AuthSchemeType.oauth2, flows={})

    # Config 1: Real secret
    auth_credential_1 = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="client_id", client_secret="real_secret", auth_uri="uri"
        ),
    )
    config1 = AuthConfig(
        auth_scheme=auth_scheme, raw_auth_credential=auth_credential_1
    )

    # Config 2: Redacted secret
    auth_credential_2 = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id="client_id", client_secret="<redacted>", auth_uri="uri"
        ),
    )
    config2 = AuthConfig(
        auth_scheme=auth_scheme, raw_auth_credential=auth_credential_2
    )

    # Keys should be identical
    key1 = config1.credential_key
    key2 = config2.credential_key

    self.assertEqual(key1, key2, f"Keys should match! {key1} vs {key2}")
