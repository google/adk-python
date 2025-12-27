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

from pydantic import model_validator

from ...features import experimental
from ...features import FeatureName
from .._google_credentials import BaseGoogleCredentialsConfig

PUBSUB_TOKEN_CACHE_KEY = "pubsub_token_cache"
PUBSUB_DEFAULT_SCOPE = ("https://www.googleapis.com/auth/pubsub",)


@experimental(FeatureName.GOOGLE_CREDENTIALS_CONFIG)
class PubSubCredentialsConfig(BaseGoogleCredentialsConfig):
  """Pub/Sub Credentials Configuration for Google API tools (Experimental).

  Please do not use this in production, as it may be deprecated later.
  """

  @model_validator(mode="after")
  def __post_init__(self) -> PubSubCredentialsConfig:
    """Populate default scope if scopes is None."""
    super().__post_init__()

    if not self.scopes:
      self.scopes = PUBSUB_DEFAULT_SCOPE

    # Set the token cache key
    self._token_cache_key = PUBSUB_TOKEN_CACHE_KEY

    return self
