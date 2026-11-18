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

"""Typed construction boundary for unannotated Google SDK classes."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import cast
from typing import Protocol
from typing import TYPE_CHECKING

from google.auth.credentials import Credentials
from google.oauth2 import credentials as user_credentials
from google.oauth2 import service_account

if TYPE_CHECKING:
  from google.api_core.client_info import ClientInfo
  from google.api_core.gapic_v1.client_info import ClientInfo as GapicClientInfo


class _ApiRepresentable(Protocol):

  def to_api_repr(self) -> dict[str, object]:
    ...


class _ClientInfoFactory(Protocol):

  def __call__(self, *, user_agent: str) -> ClientInfo:
    ...


class _GapicClientInfoFactory(Protocol):

  def __call__(self, *, user_agent: str) -> GapicClientInfo:
    ...


class _ServiceAccountCredentialsFactory(Protocol):

  def __call__(self, info: Mapping[str, object]) -> Credentials:
    ...


class _UserCredentialsFactory(Protocol):

  def __call__(self, *, token: str) -> user_credentials.Credentials:
    ...


def read_api_repr(obj: object) -> dict[str, object]:
  """Read the API representation of an unannotated SDK object."""
  return cast(_ApiRepresentable, obj).to_api_repr()


def create_client_info(*, user_agent: str) -> ClientInfo:
  """Create client metadata through the SDK's unannotated constructor."""
  # Deferred: google-api-core ships only in the `gcp` extra.
  from google.api_core.client_info import ClientInfo

  factory = cast(_ClientInfoFactory, ClientInfo)
  return factory(user_agent=user_agent)


def create_gapic_client_info(*, user_agent: str) -> GapicClientInfo:
  """Create GAPIC client metadata through its unannotated constructor."""
  # Deferred: importing gapic_v1 at module scope would require grpc.
  from google.api_core.gapic_v1 import client_info as gapic_client_info

  factory = cast(_GapicClientInfoFactory, gapic_client_info.ClientInfo)
  return factory(user_agent=user_agent)


def load_service_account_credentials(raw_json: str) -> Credentials:
  """Parse service-account JSON and construct typed credentials."""
  try:
    info: object = json.loads(raw_json)
  except json.JSONDecodeError as e:
    raise ValueError(f"Invalid service account JSON: {e}") from e
  if not isinstance(info, dict) or not all(
      isinstance(key, str) for key in info
  ):
    raise ValueError("Service account JSON must contain an object.")

  factory = cast(
      _ServiceAccountCredentialsFactory,
      service_account.Credentials.from_service_account_info,
  )
  return factory(cast(dict[str, object], info))


def create_user_credentials(*, token: str) -> user_credentials.Credentials:
  """Create OAuth user credentials through the unannotated constructor."""
  factory = cast(_UserCredentialsFactory, user_credentials.Credentials)
  return factory(token=token)
