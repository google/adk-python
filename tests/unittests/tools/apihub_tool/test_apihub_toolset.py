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

from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools.apihub_tool.apihub_toolset import APIHubToolset
from google.adk.tools.apihub_tool.clients.apihub_client import BaseAPIHubClient
import pytest
import yaml


class MockAPIHubClient(BaseAPIHubClient):

  def get_spec_content(self, _apihub_resource_name: str) -> str:
    return """
openapi: 3.0.0
info:
  version: 1.0.0
  title: Mock API
  description: Mock API Description
paths:
  /test:
    get:
      summary: Test GET endpoint
      operationId: testGet
      responses:
        '200':
          description: Successful response
    """


# Fixture for a basic APIHubToolset
@pytest.fixture
def basic_apihub_toolset():
  apihub_client = MockAPIHubClient()
  tool = APIHubToolset(
      apihub_resource_name='test_resource', apihub_client=apihub_client
  )
  return tool


# Fixture for an APIHubToolset with lazy loading
@pytest.fixture
def lazy_apihub_toolset():
  apihub_client = MockAPIHubClient()
  tool = APIHubToolset(
      apihub_resource_name='test_resource',
      apihub_client=apihub_client,
      lazy_load_spec=True,
  )
  return tool


# Fixture for auth scheme
@pytest.fixture
def mock_auth_scheme():
  return MagicMock(spec=AuthScheme)


# Fixture for auth credential
@pytest.fixture
def mock_auth_credential():
  return MagicMock(spec=AuthCredential)


# Test cases
@pytest.mark.asyncio
async def test_apihub_toolset_initialization(basic_apihub_toolset):
  assert basic_apihub_toolset.name == 'mock_api'
  assert basic_apihub_toolset.description == 'Mock API Description'
  assert basic_apihub_toolset._apihub_resource_name == 'test_resource'
  assert not basic_apihub_toolset._lazy_load_spec
  generated_tools = await basic_apihub_toolset.get_tools()
  assert len(generated_tools) == 1
  assert 'test_get' == generated_tools[0].name


@pytest.mark.asyncio
async def test_apihub_toolset_lazy_loading(lazy_apihub_toolset):
  assert lazy_apihub_toolset._lazy_load_spec
  generated_tools = await lazy_apihub_toolset.get_tools()
  assert generated_tools

  tools = await lazy_apihub_toolset.get_tools()
  assert len(tools) == 1
  'test_get' == tools[0].name


def test_apihub_toolset_no_title_in_spec(basic_apihub_toolset):
  spec = """
openapi: 3.0.0
info:
  version: 1.0.0
paths:
  /empty_desc_test:
    delete:
      summary: Test DELETE endpoint
      operationId: emptyDescTest
      responses:
        '200':
          description: Successful response
    """

  class MockAPIHubClientEmptySpec(BaseAPIHubClient):

    def get_spec_content(self, _apihub_resource_name: str) -> str:
      return spec

  apihub_client = MockAPIHubClientEmptySpec()
  toolset = APIHubToolset(
      apihub_resource_name='test_resource',
      apihub_client=apihub_client,
  )

  assert toolset.name == 'unnamed'


def test_apihub_toolset_empty_description_in_spec():
  spec = """
openapi: 3.0.0
info:
  version: 1.0.0
  title: Empty Description API
paths:
  /empty_desc_test:
    delete:
      summary: Test DELETE endpoint
      operationId: emptyDescTest
      responses:
        '200':
          description: Successful response
    """

  class MockAPIHubClientEmptySpec(BaseAPIHubClient):

    def get_spec_content(self, _apihub_resource_name: str) -> str:
      return spec

  apihub_client = MockAPIHubClientEmptySpec()
  toolset = APIHubToolset(
      apihub_resource_name='test_resource',
      apihub_client=apihub_client,
  )

  assert toolset.name == 'empty_description_api'
  assert toolset.description == ''


@pytest.mark.asyncio
async def test_get_tools_with_auth(mock_auth_scheme, mock_auth_credential):
  apihub_client = MockAPIHubClient()
  tool = APIHubToolset(
      apihub_resource_name='test_resource',
      apihub_client=apihub_client,
      auth_scheme=mock_auth_scheme,
      auth_credential=mock_auth_credential,
  )
  tools = await tool.get_tools()
  assert len(tools) == 1


@pytest.mark.asyncio
async def test_apihub_toolset_get_tools_lazy_load_empty_spec():

  class MockAPIHubClientEmptySpec(BaseAPIHubClient):

    def get_spec_content(self, _apihub_resource_name: str) -> str:
      return ''

  apihub_client = MockAPIHubClientEmptySpec()
  tool = APIHubToolset(
      apihub_resource_name='test_resource',
      apihub_client=apihub_client,
      lazy_load_spec=True,
  )
  tools = await tool.get_tools()
  assert not tools


@pytest.mark.asyncio
async def test_apihub_toolset_get_tools_invalid_yaml():

  class MockAPIHubClientInvalidYAML(BaseAPIHubClient):

    def get_spec_content(self, _apihub_resource_name: str) -> str:
      return '{invalid yaml'  # Return invalid YAML

  with pytest.raises(yaml.YAMLError):
    apihub_client = MockAPIHubClientInvalidYAML()
    tool = APIHubToolset(
        apihub_resource_name='test_resource',
        apihub_client=apihub_client,
    )
    await tool.get_tools()


if __name__ == '__main__':
  pytest.main([__file__])
