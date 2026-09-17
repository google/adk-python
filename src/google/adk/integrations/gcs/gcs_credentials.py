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

from ...features import experimental
from ...features import FeatureName
from ...tools._google_credentials import BaseGoogleCredentialsConfig

GCS_TOKEN_CACHE_KEY = "gcs_token_cache"
GCS_DEFAULT_SCOPE = [
    "https://www.googleapis.com/auth/devstorage.full_control",
]


@experimental(FeatureName.GOOGLE_CREDENTIALS_CONFIG)
class GCSCredentialsConfig(BaseGoogleCredentialsConfig):
  """GCS Credentials Configuration for Google API tools (Experimental)."""

  def __post_init__(self) -> GCSCredentialsConfig:
    """Populate default scope if scopes is None."""
    super().__post_init__()

    if not self.scopes:
      self.scopes = GCS_DEFAULT_SCOPE

    # Set the token cache key
    self._token_cache_key = GCS_TOKEN_CACHE_KEY

    return self
