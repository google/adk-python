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

import json
import logging
from typing import List
from typing import Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel
from pydantic import ValidationError

from ..utils.feature_decorator import experimental

logger = logging.getLogger("google_adk." + __name__)


@experimental
class AuthorizationServerMetadata(BaseModel):
  """Represents the OAuth2 authorization server metadata per RFC8414."""

  issuer: str
  authorization_endpoint: str
  token_endpoint: str
  scopes_supported: Optional[List[str]] = None
  registration_endpoint: Optional[str] = None


@experimental
class ProtectedResourceMetadata(BaseModel):
  """Represents the OAuth2 protected resource metadata per RFC9728."""

  resource: str
  authorization_servers: List[str] = []


@experimental
class OAuth2DiscoveryManager:
  """Implements Metadata discovery for OAuth2 following RFC8414 and RFC9728."""

  async def discover_auth_server_metadata(
      self, issuer_url: str
  ) -> Optional[AuthorizationServerMetadata]:
    """Discovers the OAuth2 authorization server metadata."""
    try:
      parsed_url = urlparse(issuer_url)
      base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
      path = parsed_url.path
    except ValueError as e:
      logger.warning("Failed to parse issuer_url %s: %s", issuer_url, e)
      return None

    # Try the standard well-known endpoints in order.
    if path and path != "/":
      endpoints_to_try = [
          # 1. OAuth 2.0 Authorization Server Metadata with path insertion
          f"{base_url}/.well-known/oauth-authorization-server{path}",
          # 2. OpenID Connect Discovery 1.0 with path insertion
          f"{base_url}/.well-known/openid-configuration{path}",
          # 3. OpenID Connect Discovery 1.0 with path appending
          f"{base_url}{path}/.well-known/openid-configuration",
      ]
    else:
      endpoints_to_try = [
          # 1. OAuth 2.0 Authorization Server Metadata
          f"{base_url}/.well-known/oauth-authorization-server",
          # 2. OpenID Connect Discovery 1.0
          f"{base_url}/.well-known/openid-configuration",
      ]

    async with httpx.AsyncClient() as client:
      for endpoint in endpoints_to_try:
        try:
          response = await client.get(endpoint, timeout=5)
          response.raise_for_status()
          metadata = AuthorizationServerMetadata.model_validate(response.json())
          # Validate issuer to defend against MIX-UP attacks
          if metadata.issuer == issuer_url.rstrip("/"):
            return metadata
          else:
            logger.warning(
                "Issuer in metadata %s does not match issuer_url %s",
                metadata.issuer,
                issuer_url,
            )
        except httpx.HTTPError as e:
          logger.debug("Failed to fetch metadata from %s: %s", endpoint, e)
        except (json.decoder.JSONDecodeError, ValidationError) as e:
          logger.debug("Failed to parse metadata from %s: %s", endpoint, e)
    return None

  async def discover_resource_metadata(
      self, resource_url: str
  ) -> Optional[ProtectedResourceMetadata]:
    """Discovers the OAuth2 protected resource metadata."""
    try:
      parsed_url = urlparse(resource_url)
      base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
      path = parsed_url.path
    except ValueError as e:
      logger.warning("Failed to parse resource_url %s: %s", resource_url, e)
      return None

    if path and path != "/":
      well_known_endpoint = (
          f"{base_url}/.well-known/oauth-protected-resource{path}"
      )
    else:
      well_known_endpoint = f"{base_url}/.well-known/oauth-protected-resource"

    async with httpx.AsyncClient() as client:
      try:
        response = await client.get(well_known_endpoint, timeout=5)
        response.raise_for_status()
        metadata = ProtectedResourceMetadata.model_validate(response.json())
        # Validate resource to defend against MIX-UP attacks
        if metadata.resource == resource_url.rstrip("/"):
          return metadata
        else:
          logger.warning(
              "Resource in metadata %s does not match resource_url %s",
              metadata.resource,
              resource_url,
          )
      except httpx.HTTPError as e:
        logger.debug(
            "Failed to fetch metadata from %s: %s", well_known_endpoint, e
        )
      except (json.decoder.JSONDecodeError, ValidationError) as e:
        logger.debug(
            "Failed to parse metadata from %s: %s", well_known_endpoint, e
        )

    return None
