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

"""Utilities for mTLS regional endpoint resolution."""

from __future__ import annotations

import enum
import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from google.auth.transport import mtls

if TYPE_CHECKING:
  import requests

logger = logging.getLogger("google_adk." + __name__)

_GOOGLEAPIS_SUFFIX = ".googleapis.com"
_MTLS_GOOGLEAPIS_SUFFIX = ".mtls.googleapis.com"


class MtlsEndpoint(enum.Enum):
  """Enum for the mTLS endpoint setting."""

  AUTO = "auto"
  ALWAYS = "always"
  NEVER = "never"


def _mtls_endpoint_setting() -> MtlsEndpoint:
  """Returns the GOOGLE_API_USE_MTLS_ENDPOINT setting, defaulting to AUTO."""
  setting = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", MtlsEndpoint.AUTO.value
  ).lower()
  try:
    return MtlsEndpoint(setting)
  except ValueError:
    return MtlsEndpoint.AUTO


def use_client_cert_effective() -> bool:
  """Returns whether client certificate should be used for mTLS."""
  try:
    return bool(mtls.should_use_client_cert())
  except (ImportError, AttributeError):
    return (
        os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false").lower()
        == "true"
    )


def get_api_endpoint(
    location: str, default_template: str, mtls_template: str
) -> str:
  """Returns API endpoint based on mTLS configuration and cert availability.

  Args:
      location: The region location.
      default_template: Template for default regional endpoint (e.g.
        "secretmanager.{location}.rep.googleapis.com").
      mtls_template: Template for mTLS regional endpoint (e.g.
        "secretmanager.{location}.rep.mtls.googleapis.com").
  """
  use_mtls_endpoint = _mtls_endpoint_setting()
  if (use_mtls_endpoint == MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint == MtlsEndpoint.AUTO and use_client_cert_effective()
  ):
    return mtls_template.format(location=location)
  return default_template.format(location=location)


def is_non_mtls_googleapis_endpoint(url: str) -> bool:
  """Returns whether url points at a *.googleapis.com host without the mTLS infix."""
  if not url:
    return False
  host = urlsplit(url).hostname or ""
  return (
      host.endswith(_GOOGLEAPIS_SUFFIX) and _MTLS_GOOGLEAPIS_SUFFIX not in host
  )


def effective_googleapis_endpoint(url: str) -> str:
  """Rewrites a *.googleapis.com url to its .mtls.googleapis.com variant.

  Honors GOOGLE_API_USE_MTLS_ENDPOINT=never as an opt-out. Hosts that are not
  googleapis.com hosts, or are already mTLS hosts, are returned unchanged so
  non-Google providers are never affected.
  """
  if not is_non_mtls_googleapis_endpoint(url):
    return url
  if _mtls_endpoint_setting() == MtlsEndpoint.NEVER:
    return url
  parsed = urlsplit(url)
  host = parsed.hostname or ""
  new_host = host[: -len(_GOOGLEAPIS_SUFFIX)] + _MTLS_GOOGLEAPIS_SUFFIX
  netloc = f"{new_host}:{parsed.port}" if parsed.port else new_host
  return urlunsplit(
      (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
  )


def configure_session_for_mtls(session: requests.Session) -> bool:
  """Mounts a mutual-TLS adapter on a requests session when a client cert exists.

  authlib's OAuth2Session is a requests.Session but not a google-auth
  AuthorizedSession, so it lacks configure_mtls_channel(). This replicates that
  method's effect: load the application-default client certificate and mount an
  adapter that presents it on https connections.

  Returns True if a client certificate was found and the adapter was mounted.
  """
  try:
    from google.auth import exceptions as ga_exceptions
    from google.auth.transport import _mtls_helper
    from google.auth.transport.requests import _MutualTlsAdapter
  except ImportError:
    return False

  cert_source = (
      mtls.default_client_cert_source()
      if mtls.has_default_client_cert_source()
      else None
  )
  try:
    is_mtls, cert, key = _mtls_helper.get_client_cert_and_key(cert_source)
  except (ImportError, ga_exceptions.GoogleAuthError) as e:
    logger.warning(
        "Could not load client certificate for mTLS; falling back to non-mTLS"
        " token request: %s",
        e,
    )
    return False

  if is_mtls:
    session.mount("https://", _MutualTlsAdapter(cert, key))
  return bool(is_mtls)


class MtlsClientCerts:
  """Manages the creation and lifecycle of client certificates for mTLS.

  Extracts certificates to a temporary directory that is automatically cleaned up
  when the instance is garbage collected.
  """

  def __init__(self) -> None:
    self._tempdir: tempfile.TemporaryDirectory[str] | None = None
    self.cert_path: str | None = None
    self.key_path: str | None = None
    self.passphrase: bytes | None = None
    self._lock = threading.Lock()
    self._initialized = False

  def get_certs(self) -> tuple[str | None, str | None, bytes | None]:
    """Extracts and returns the certificate paths and passphrase.

    Returns:
        A tuple of (cert_path, key_path, passphrase) if client certificates
        are available, otherwise (None, None, None).
    """
    with self._lock:
      if self._initialized:
        return self.cert_path, self.key_path, self.passphrase

      if not mtls.has_default_client_cert_source():
        self._initialized = True
        return None, None, None

      self._tempdir = tempfile.TemporaryDirectory()
      cert_path_tmp = os.path.join(self._tempdir.name, "cert.pem")
      key_path_tmp = os.path.join(self._tempdir.name, "key.pem")

      try:
        cert_source = mtls.default_client_encrypted_cert_source(
            cert_path_tmp, key_path_tmp
        )
        _, _, passphrase = cert_source()
      except Exception as e:
        # If extraction fails, we should fail loud.
        self._tempdir.cleanup()
        self._tempdir = None
        raise RuntimeError(
            f"Failed to extract default client certificates for mTLS: {e}"
        ) from e

      self.cert_path = cert_path_tmp
      self.key_path = key_path_tmp
      self.passphrase = passphrase
      self._initialized = True

      return self.cert_path, self.key_path, self.passphrase

  def close(self) -> None:
    """Manually cleans up the temporary directory."""
    with self._lock:
      if self._tempdir:
        self._tempdir.cleanup()
        self._tempdir = None
      self.cert_path = None
      self.key_path = None
      self.passphrase = None
      self._initialized = False
