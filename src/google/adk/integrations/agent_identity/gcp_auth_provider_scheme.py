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

from typing import List
from typing import Literal
from typing import Optional

from google.adk.auth.auth_schemes import CustomAuthScheme
from pydantic import Field


class GcpAuthProviderScheme(CustomAuthScheme):
  """The Agent Identity authentication scheme for Google Cloud Platform.

  Attributes:
    name: The name of the GCP Auth Provider resource to use.
    scopes: Optional. A list of OAuth2 scopes to request.
    continue_uri: Optional. A type of redirect URI. It is distinct from the
      standard OAuth2 redirect URI. Its purpose is to reauthenticate the user to
      prevent phishing attacks and to finalize the managed OAuth flow. The
      standard, Google-hosted OAuth2 redirect URI will redirect the user to this
      continue URI. The agent will include this URI in every 3-legged OAuth
      request sent to the upstream Agent Identity Credential service. Developers
      must ensure this URI is hosted (e.g. on GCP, a third-party cloud,
      on-prem), preferably alongside the agent client's web server.
      TODO: Add public documentation link for more information once available.
    type_: The type of the security scheme, always "gcpAuthProviderScheme".
  """

  type_: Literal["gcpAuthProviderScheme"] = Field(
      default="gcpAuthProviderScheme", alias="type"
  )
  name: str
  scopes: Optional[List[str]] = None
  continue_uri: Optional[str] = None
