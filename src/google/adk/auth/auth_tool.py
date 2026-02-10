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

import hashlib
import json
from typing import Optional

from pydantic import BaseModel
from typing_extensions import deprecated

from .auth_credential import AuthCredential
from .auth_credential import BaseModelWithConfig
from .auth_schemes import AuthScheme


def _stable_model_digest(model: BaseModel) -> str:
  """Returns a stable digest for a pydantic model.

  The digest is stable across:
  - Python hash seeds (does not use `hash()`).
  - Dict insertion ordering differences (canonicalizes via `sort_keys=True`).
  - Pydantic `model_extra` values (ignored).
  """
  if getattr(model, "model_extra", None):
    model = model.model_copy(deep=True)
    model.model_extra.clear()

  dumped = model.model_dump(by_alias=True, exclude_none=True, mode="json")
  canonical_json = json.dumps(
      dumped,
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
  )
  return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


class AuthConfig(BaseModelWithConfig):
  """The auth config sent by tool asking client to collect auth credentials and

  adk and client will help to fill in the response
  """

  auth_scheme: AuthScheme
  """The auth scheme used to collect credentials"""
  raw_auth_credential: Optional[AuthCredential] = None
  """The raw auth credential used to collect credentials. The raw auth
  credentials are used in some auth scheme that needs to exchange auth
  credentials. e.g. OAuth2 and OIDC. For other auth scheme, it could be None.
  """
  exchanged_auth_credential: Optional[AuthCredential] = None
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

  credential_key: Optional[str] = None
  """A user specified key used to load and save this credential in a credential
  service.
  """

  def __init__(self, **data):
    super().__init__(**data)
    if self.credential_key:
      return
    for obj in (self.raw_auth_credential, self.auth_scheme):
      if not obj or not getattr(obj, "model_extra", None):
        continue
      for key in ("credential_key", "credentialKey"):
        value = obj.model_extra.get(key)
        if isinstance(value, str) and value:
          self.credential_key = value
          return
    self.credential_key = self.get_credential_key()

  @deprecated("This method is deprecated. Use credential_key instead.")
  def get_credential_key(self):
    """Builds a stable key based on auth_scheme and raw_auth_credential.

    This is used to save/load credentials to/from a credential service when
    `credential_key` is not explicitly provided.
    """

    auth_scheme = self.auth_scheme

    if auth_scheme.model_extra:
      auth_scheme = auth_scheme.model_copy(deep=True)
      auth_scheme.model_extra.clear()
    scheme_name = (
        f"{auth_scheme.type_.name}_{_stable_model_digest(auth_scheme)}"
        if auth_scheme
        else ""
    )

    auth_credential = self.raw_auth_credential
    if auth_credential and auth_credential.model_extra:
      auth_credential = auth_credential.model_copy(deep=True)
      auth_credential.model_extra.clear()
    if auth_credential and auth_credential.oauth2:
      auth_credential = auth_credential.model_copy(deep=True)
      auth_credential.oauth2.auth_uri = None
      auth_credential.oauth2.state = None
      auth_credential.oauth2.auth_response_uri = None
      auth_credential.oauth2.auth_code = None
      auth_credential.oauth2.access_token = None
      auth_credential.oauth2.refresh_token = None
      auth_credential.oauth2.expires_at = None
      auth_credential.oauth2.expires_in = None
    credential_name = (
        f"{auth_credential.auth_type.value}_{_stable_model_digest(auth_credential)}"
        if auth_credential
        else ""
    )

    return f"adk_{scheme_name}_{credential_name}"


class AuthToolArguments(BaseModelWithConfig):
  """the arguments for the special long running function tool that is used to

  request end user credentials.
  """

  function_call_id: str
  auth_config: AuthConfig
