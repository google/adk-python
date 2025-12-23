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

from functools import cached_property
import logging
import os
from typing import Optional
from typing import TYPE_CHECKING

from google.adk import version as adk_version
from google.genai import types
from typing_extensions import override

from ..utils.env_utils import is_env_enabled
from .google_llm import Gemini

if TYPE_CHECKING:
  from google.genai import Client

  from .llm_request import LlmRequest


logger = logging.getLogger('google_adk.' + __name__)

_APIGEE_PROXY_URL_ENV_VARIABLE_NAME = 'APIGEE_PROXY_URL'
_GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME = 'GOOGLE_GENAI_USE_VERTEXAI'
_PROJECT_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_PROJECT'
_LOCATION_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_LOCATION'


class ApigeeLlm(Gemini):
  """A BaseLlm implementation for calling Apigee proxy.

  Attributes:
    model: The name of the Gemini model.
  """

  def __init__(
      self,
      *,
      model: str,
      proxy_url: str | None = None,
      custom_headers: dict[str, str] | None = None,
      retry_options: Optional[types.HttpRetryOptions] = None,
  ):
    """Initializes the Apigee LLM backend.

    Args:
      model: The model string specifies the LLM provider (e.g., Vertex AI,
        Gemini), API version, and the model ID. Supported format:
        `apigee/[<provider>/][<version>/]<model_id>`

        Components
          `provider` (optional): `vertex_ai` or `gemini`. If omitted, behavior
            depends on the `GOOGLE_GENAI_USE_VERTEXAI` environment variable. If
            that is not set to TRUE or 1, it defaults to `gemini`. `provider`
            takes precedence over `GOOGLE_GENAI_USE_VERTEXAI`.
          `version` (optional): The API version (e.g., `v1`, `v1beta`). If
            omitted, the default version for the provider is used.
          `model_id` (required): The model identifier (e.g.,
            `gemini-2.5-flash`).

        Examples
          - `apigee/gemini-2.5-flash`
          - `apigee/v1/gemini-2.5-flash`
          - `apigee/vertex_ai/gemini-2.5-flash`
          - `apigee/gemini/v1/gemini-2.5-flash`
          - `apigee/vertex_ai/v1beta/gemini-2.5-flash`

      proxy_url: The URL of the Apigee proxy.
      custom_headers: A dictionary of headers to be sent with the request.
      retry_options: Allow google-genai to retry failed responses.
    """

    super().__init__(model=model, retry_options=retry_options)
    # Validate the model string. Create a helper method to validate the model
    # string.
    if not _validate_model_string(model):
      raise ValueError(f'Invalid model string: {model}')

    self._isvertexai = _identify_vertexai(model)

    # Set the project and location for Vertex AI.
    if self._isvertexai:
      self._project = os.environ.get(_PROJECT_ENV_VARIABLE_NAME)
      self._location = os.environ.get(_LOCATION_ENV_VARIABLE_NAME)

      if not self._project:
        raise ValueError(
            f'The {_PROJECT_ENV_VARIABLE_NAME} environment variable must be'
            ' set.'
        )

      if not self._location:
        raise ValueError(
            f'The {_LOCATION_ENV_VARIABLE_NAME} environment variable must be'
            ' set.'
        )

    self._api_version = _identify_api_version(model)
    self._proxy_url = proxy_url or os.environ.get(
        _APIGEE_PROXY_URL_ENV_VARIABLE_NAME
    )
    self._custom_headers = custom_headers or {}
    self._user_agent = f'google-adk/{adk_version.__version__}'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'apigee\/.*',
    ]

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    from google.genai import Client

    kwargs_for_http_options = {}
    if self._api_version:
      kwargs_for_http_options['api_version'] = self._api_version
    http_options = types.HttpOptions(
        base_url=self._proxy_url,
        headers=self._merge_tracking_headers(self._custom_headers),
        retry_options=self.retry_options,
        **kwargs_for_http_options,
    )

    kwargs_for_client = {}
    kwargs_for_client['vertexai'] = self._isvertexai
    if self._isvertexai:
      kwargs_for_client['project'] = self._project
      kwargs_for_client['location'] = self._location

    return Client(
        http_options=http_options,
        **kwargs_for_client,
    )

  @override
  async def _preprocess_request(self, llm_request: LlmRequest) -> None:
    llm_request.model = _get_model_id(llm_request.model)
    await super()._preprocess_request(llm_request)


def _identify_vertexai(model: str) -> bool:
  """Returns True if the model spec starts with apigee/vertex_ai."""
  return not model.startswith('apigee/gemini/') and (
      model.startswith('apigee/vertex_ai/')
      or is_env_enabled(_GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME)
  )


def _identify_api_version(model: str) -> str:
  """Returns the api version for the model spec."""
  model = model.removeprefix('apigee/')
  components = model.split('/')

  if len(components) == 3:
    # Format: <provider>/<version>/<model_id>
    return components[1]
  if len(components) == 2:
    # Format: <version>/<model_id> or <provider>/<model_id>
    # _validate_model_string ensures that if the first component is not a
    # provider, it can be a version.
    if components[0] not in ('vertex_ai', 'gemini') and components[
        0
    ].startswith('v'):
      return components[0]
  return ''


def _get_model_id(model: str) -> str:
  """Returns the model ID for the model spec."""
  model = model.removeprefix('apigee/')
  components = model.split('/')

  # Model_id is the last component in the model string.
  return components[-1]


def _validate_model_string(model: str) -> bool:
  """Validates the model string for Apigee LLM.

  The model string specifies the LLM provider (e.g., Vertex AI, Gemini), API
  version, and the model ID.

  Args:
    model: The model string. Supported format:
      `apigee/[<provider>/][<version>/]<model_id>`

  Returns:
    True if the model string is valid, False otherwise.
  """
  if not model.startswith('apigee/'):
    return False

  # Remove leading "apigee/" from the model string.
  model = model.removeprefix('apigee/')

  # The string has to be non-empty. i.e. the model_id cannot be empty.
  if not model:
    return False

  components = model.split('/')
  # If the model string has exactly 1 component, it means only the model_id is
  # present. This is a valid format.
  if len(components) == 1:
    return True

  # If the model string has more than 3 components, it is invalid.
  if len(components) > 3:
    return False

  # If the model string has 3 components, it means only the provider, version,
  # and model_id are present. This is a valid format.
  if len(components) == 3:
    # Format: <provider>/<version>/<model_id>
    if components[0] not in ('vertex_ai', 'gemini'):
      return False
    if not components[1].startswith('v'):
      return False
    return True

  # If the model string has 2 components, it means either the provider or the
  # version (but not both), and model_id are present.
  if len(components) == 2:
    if components[0] in ['vertex_ai', 'gemini']:
      return True
    if components[0].startswith('v'):
      return True
    return False

  return False
