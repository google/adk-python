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

from __future__ import annotations

from ...utils.feature_decorator import experimental
from .._google_credentials import BaseGoogleCredentialsConfig

BIGQUERY_TOKEN_CACHE_KEY = "bigquery_token_cache"
BIGQUERY_DEFAULT_SCOPE = ["https://www.googleapis.com/auth/bigquery"]


@experimental
class BigQueryCredentialsConfig(BaseGoogleCredentialsConfig):
  """BigQuery Credentials Configuration for Google API tools (Experimental).

  Please do not use this in production, as it may be deprecated later.
  """

  def __post_init__(self) -> BigQueryCredentialsConfig:
    """Populate default scope if scopes is None."""
    super().__post_init__()

    if not self.scopes:
      self.scopes = BIGQUERY_DEFAULT_SCOPE

    # Set the token cache key
    self._token_cache_key = BIGQUERY_TOKEN_CACHE_KEY

    return self
