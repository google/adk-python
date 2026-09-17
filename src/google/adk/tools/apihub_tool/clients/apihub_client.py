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

from abc import ABC
from abc import abstractmethod
import base64
from collections.abc import Mapping
from collections.abc import Sequence
import json
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Protocol
from urllib.parse import parse_qs
from urllib.parse import urlparse

from google.auth import default as default_service_credential
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests

from ....utils import _mtls_utils

_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30


class _ServiceAccountCredentialsFactory(Protocol):

  def __call__(
      self,
      info: Mapping[str, object],
      *,
      scopes: Sequence[str] | None = None,
  ) -> Credentials:
    ...


def _response_object(response: requests.Response) -> dict[str, object]:
  """Validate the JSON object returned by API Hub."""
  payload: object = response.json()
  if not isinstance(payload, dict) or not all(
      isinstance(key, str) for key in payload
  ):
    raise ValueError("API Hub returned a non-object JSON response.")
  return cast(dict[str, object], payload)


def _string_list(value: object, *, field: str) -> list[str]:
  """Validate a JSON array whose members must all be strings."""
  if not isinstance(value, list) or not all(
      isinstance(item, str) for item in value
  ):
    raise ValueError(f"API Hub field '{field}' must be a list of strings.")
  return cast(list[str], value)


def _object_list(value: object, *, field: str) -> list[dict[str, object]]:
  """Validate a JSON array whose members must be string-keyed objects."""
  if not isinstance(value, list):
    raise ValueError(f"API Hub field '{field}' must be a list of objects.")

  result: list[dict[str, object]] = []
  for item in value:
    if not isinstance(item, dict) or not all(
        isinstance(key, str) for key in item
    ):
      raise ValueError(f"API Hub field '{field}' must be a list of objects.")
    result.append(cast(dict[str, object], item))
  return result


class BaseAPIHubClient(ABC):
  """Base class for API Hub clients."""

  @abstractmethod
  def get_spec_content(self, resource_name: str) -> str:
    """From a given resource name, get the spec in the API Hub."""
    raise NotImplementedError()


class APIHubClient(BaseAPIHubClient):
  """Client for interacting with the API Hub service."""

  def __init__(
      self,
      *,
      access_token: str | None = None,
      service_account_json: str | None = None,
  ):
    """Initializes the APIHubClient.

    You must set either access_token or service_account_json. This
    credential is used for sending request to API Hub API.

    Args:
        access_token: Google Access token. Generate with gcloud cli `gcloud auth
          print-access-token`. Useful for local testing.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential.
    """
    self.root_url = "https://apihub.googleapis.com/v1"
    self.credential_cache: Credentials | None = None
    self.access_token: str | None = None
    self.service_account: str | None = None

    if access_token:
      self.access_token = access_token
    elif service_account_json:
      self.service_account = service_account_json

  def get_spec_content(self, path: str) -> str:
    """From a given path, get the first spec available in the API Hub.

    - If path includes /apis/apiname, get the first spec of that API
    - If path includes /apis/apiname/versions/versionname, get the first spec
      of that API Version
    - If path includes /apis/apiname/versions/versionname/specs/specname, return
      that spec

    Path can be resource name (projects/xxx/locations/us-central1/apis/apiname),
    and URL from the UI
    (https://console.cloud.google.com/apigee/api-hub/apis/apiname?project=xxx)

    Args:
        path: The path to the API, API Version, or API Spec.

    Returns:
        The content of the first spec available in the API Hub.
    """
    apihub_resource_name, api_version_resource_name, api_spec_resource_name = (
        self._extract_resource_name(path)
    )

    if apihub_resource_name and not api_version_resource_name:
      api = self.get_api(apihub_resource_name)
      versions = _string_list(api.get("versions", []), field="versions")
      if not versions:
        raise ValueError(
            f"No versions found in API Hub resource: {apihub_resource_name}"
        )
      api_version_resource_name = versions[0]

    if api_version_resource_name and not api_spec_resource_name:
      api_version = self.get_api_version(api_version_resource_name)
      spec_resource_names = _string_list(
          api_version.get("specs", []), field="specs"
      )
      if not spec_resource_names:
        raise ValueError(
            f"No specs found in API Hub version: {api_version_resource_name}"
        )
      api_spec_resource_name = spec_resource_names[0]

    if api_spec_resource_name:
      spec_content = self._fetch_spec(api_spec_resource_name)
      return spec_content

    raise ValueError(f"No API Hub resource found in path: {path}")

  # The three accessors below keep their released Dict[str, Any] returns. The
  # bodies now validate the payload, but narrowing Any to object here would
  # stop callers indexing into the result.
  def list_apis(self, project: str, location: str) -> List[Dict[str, Any]]:
    """Lists all APIs in the specified project and location.

    Args:
        project: The Google Cloud project name.
        location: The location of the API Hub resources (e.g., 'us-central1').

    Returns:
        A list of API dictionaries, or an empty list if an error occurs.
    """
    url = f"{self.root_url}/projects/{project}/locations/{location}/apis"
    response = self._get(url)
    payload = _response_object(response)
    return _object_list(payload.get("apis", []), field="apis")

  def get_api(self, api_resource_name: str) -> Dict[str, Any]:
    """Get API detail by API name.

    Args:
        api_resource_name: Resource name of this API, like
          projects/xxx/locations/us-central1/apis/apiname

    Returns:
        An API and details in a dict.
    """
    url = f"{self.root_url}/{api_resource_name}"
    response = self._get(url)
    return _response_object(response)

  def get_api_version(self, api_version_name: str) -> Dict[str, Any]:
    """Gets details of a specific API version.

    Args:
        api_version_name: The resource name of the API version.

    Returns:
        The API version details as a dictionary, or an empty dictionary if an
        error occurs.
    """
    url = f"{self.root_url}/{api_version_name}"
    response = self._get(url)
    return _response_object(response)

  def _fetch_spec(self, api_spec_resource_name: str) -> str:
    """Retrieves the content of a specific API specification.

    Args:
        api_spec_resource_name: The resource name of the API spec.

    Returns:
        The decoded content of the specification as a string, or an empty string
        if an error occurs.
    """
    url = f"{self.root_url}/{api_spec_resource_name}:contents"
    response = self._get(url)
    content_base64 = _response_object(response).get("contents", "")
    if not isinstance(content_base64, str):
      raise ValueError("API Hub field 'contents' must be a string.")
    if content_base64:
      content_decoded = base64.b64decode(content_base64).decode("utf-8")
      return content_decoded
    else:
      return ""

  def _extract_resource_name(
      self, url_or_path: str
  ) -> tuple[str, str | None, str | None]:
    """Extracts the resource names of an API, API Version, and API Spec from a given URL or path.

    Args:
        url_or_path: The URL (UI or resource) or path string.

    Returns:
        A dictionary containing the resource names:
        {
            "api_resource_name": "projects/*/locations/*/apis/*",
            "api_version_resource_name":
            "projects/*/locations/*/apis/*/versions/*",
            "api_spec_resource_name":
            "projects/*/locations/*/apis/*/versions/*/specs/*"
        }
        or raises ValueError if extraction fails.

    Raises:
        ValueError: If the URL or path is invalid or if required components
        (project, location, api) are missing.
    """

    query_params: dict[str, list[str]] | None = None
    try:
      parsed_url = urlparse(url_or_path)
      path = parsed_url.path
      query_params = parse_qs(parsed_url.query)

      # This is a path from UI. Remove unnecessary prefix.
      if "api-hub/" in path:
        path = path.split("api-hub")[1]
    except Exception:
      path = url_or_path

    path_segments = [segment for segment in path.split("/") if segment]

    project = None
    location = None
    api_id = None
    version_id = None
    spec_id = None

    if "projects" in path_segments:
      project_index = path_segments.index("projects")
      if project_index + 1 < len(path_segments):
        project = path_segments[project_index + 1]
    elif query_params and "project" in query_params:
      project = query_params["project"][0]

    if not project:
      raise ValueError(
          "Project ID not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/projects/PROJECT_ID' in the path or 'project=PROJECT_ID' query"
          " param in the input."
      )

    if "locations" in path_segments:
      location_index = path_segments.index("locations")
      if location_index + 1 < len(path_segments):
        location = path_segments[location_index + 1]
    if not location:
      raise ValueError(
          "Location not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/location/LOCATION_ID' in the path."
      )

    if "apis" in path_segments:
      api_index = path_segments.index("apis")
      if api_index + 1 < len(path_segments):
        api_id = path_segments[api_index + 1]
    if not api_id:
      raise ValueError(
          "API id not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/apis/API_ID' in the path."
      )
    if "versions" in path_segments:
      version_index = path_segments.index("versions")
      if version_index + 1 < len(path_segments):
        version_id = path_segments[version_index + 1]

    if "specs" in path_segments:
      spec_index = path_segments.index("specs")
      if spec_index + 1 < len(path_segments):
        spec_id = path_segments[spec_index + 1]

    api_resource_name = f"projects/{project}/locations/{location}/apis/{api_id}"
    api_version_resource_name = (
        f"{api_resource_name}/versions/{version_id}" if version_id else None
    )
    api_spec_resource_name = (
        f"{api_version_resource_name}/specs/{spec_id}"
        if version_id and spec_id
        else None
    )

    return (
        api_resource_name,
        api_version_resource_name,
        api_spec_resource_name,
    )

  def _get(self, url: str) -> requests.Response:
    """Sends an authenticated GET request to API Hub.

    When a client certificate is configured, the certificate is presented on
    the connection and the request is routed to the mTLS endpoint so that
    token binding is honored.

    Args:
        url: The absolute URL to request.

    Returns:
        The successful response.
    """
    headers = {
        "accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {self._get_access_token()}",
    }
    with requests.Session() as session:
      if (
          _mtls_utils.use_client_cert_effective()
          and _mtls_utils.configure_session_for_mtls(session)
      ):
        url = _mtls_utils.effective_googleapis_endpoint(url)
      response = session.get(
          url, headers=headers, timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS
      )
    response.raise_for_status()
    return response

  def _get_access_token(self) -> str:
    """Gets the access token for the service account.

    Returns:
        The access token.
    """
    if self.access_token:
      return self.access_token

    if self.credential_cache and not self.credential_cache.expired:
      return self._require_token(self.credential_cache)

    if self.service_account:
      try:
        service_account_info: object = json.loads(self.service_account)
        if not isinstance(service_account_info, dict) or not all(
            isinstance(key, str) for key in service_account_info
        ):
          raise ValueError("Service account JSON must contain an object.")
        factory = cast(
            _ServiceAccountCredentialsFactory,
            service_account.Credentials.from_service_account_info,
        )
        credentials: Credentials | None = factory(
            cast(dict[str, object], service_account_info),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid service account JSON: {e}") from e
    else:
      try:
        credentials, _ = default_service_credential(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
      except DefaultCredentialsError:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account or an access token to API Hub"
          " client."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return self._require_token(credentials)

  @staticmethod
  def _require_token(credentials: Credentials) -> str:
    """Return a usable token from refreshed Google credentials."""
    token: object = credentials.token
    if not isinstance(token, str) or not token:
      raise ValueError("Google credentials did not provide an access token.")
    return token
