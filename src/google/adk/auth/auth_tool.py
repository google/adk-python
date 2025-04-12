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

from pydantic import BaseModel

from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme


class AuthConfig(BaseModel):
  """The auth config sent by tool asking client to collect auth credentials and

  adk and client will help to fill in the response
  """

  auth_scheme: AuthScheme
  """The auth scheme used to collect credentials"""
  raw_auth_credential: AuthCredential = None
  """The raw auth credential used to collect credentials. The raw auth
  credentials are used in some auth scheme that needs to exchange auth
  credentials. e.g. OAuth2 and OIDC. For other auth scheme, it could be None.
  """
  exchanged_auth_credential: AuthCredential = None
  """The exchanged auth credential used to collect credentials. adk and client
  will work together to fill it. For those auth scheme that doesn't need to
  exchange auth credentials, e.g. API key, service account etc. It's filled by
  client directly. For those auth scheme that need to exchange auth credentials,
  e.g. OAuth2 and OIDC, it's first filled by adk. If the raw credentials
  passed by tool only has client id and client credential, adk will help to
  generate the corresponding authorization uri and state and store the processed
  credential in this field. If the raw credentials passed by tool already has
  authorization uri, state, etc. then it's copied to this field. Client will use
  this field to guide the user through the OAuth2 flow and fill auth response in
  this field"""


class AuthToolArguments(BaseModel):
  """the arguments for the special long running function tool that is used to

  request end user credentials.
  """

  function_call_id: str
  auth_config: AuthConfig
