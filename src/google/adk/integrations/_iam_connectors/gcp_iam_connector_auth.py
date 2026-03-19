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

from enum import Enum
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ...features import FeatureName
from ...utils.feature_decorator import experimental


class ManagedAuthSchemeType(str, Enum):
  """Enum for externally managed authentication scheme types."""

  gcpIamConnectorAuth = "gcpIamConnectorAuth"


@experimental(FeatureName.GCP_IAM_CONNECTOR_AUTH)
class GcpIamConnectorAuth(BaseModel):
  """Represents a GCP IAM Connector authentication scheme.

  Attributes:
    connector_name: The name of the GCP IAM connector to use.
    scopes: Optional. A list of OAuth2 scopes to request.
    continue_uri: Optional. A type of redirect URI. It is distinct from the
      standard OAuth2 redirect URI. Its purpose is to reauthenticate the user to
      prevent phishing attacks and to finalize the managed OAuth flow. The
      standard, Google-hosted OAuth2 redirect URI will redirect the user to this
      continue URI. The agent will include this URI in every 3-legged OAuth
      request sent to the upstream IAM Connector Credential service. Developers
      must ensure this URI is hosted (e.g. on GCP, a third-party cloud,
      on-prem), preferably alongside the agent client's web server.
      TODO: Add public documentation link for more information once available.
  """

  type_: ManagedAuthSchemeType = Field(
      default=ManagedAuthSchemeType.gcpIamConnectorAuth, alias="type"
  )
  connector_name: str
  scopes: Optional[List[str]] = None
  continue_uri: Optional[str] = None
