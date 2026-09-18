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

"""Tests for the BigQuery tool config's default project and dataset ids."""

from __future__ import annotations

from typing import Any
from unittest import mock

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations.bigquery import BigQueryCredentialsConfig
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.integrations.bigquery._bigquery_tool import BigQueryTool
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.tools.tool_context import ToolContext
import pytest


@pytest.fixture(params=[True, False], ids=["json-schema", "schema-object"])
def declaration_shape(request):
  """Exercises the tests against both function declaration builders."""
  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, request.param
  ):
    yield


def sample_func(
    project_id: str, dataset_id: str, table_id: str
) -> dict[str, Any]:
  """Echoes the ids it was called with.

  Args:
      project_id (str): The Google Cloud project id.
      dataset_id (str): The BigQuery dataset id.
      table_id (str): The BigQuery table id.

  Returns:
      dict: The ids the tool was called with.
  """
  return {
      "project_id": project_id,
      "dataset_id": dataset_id,
      "table_id": table_id,
  }


def get_params(tool: BigQueryTool) -> tuple[dict[str, Any], list[str]]:
  """Reads the declared parameters of a tool.

  A declaration carries its parameters either as a Schema or as a raw JSON
  schema, depending on whether JSON_SCHEMA_FOR_FUNC_DECL is enabled, so both
  shapes are read here.

  Args:
      tool: The tool to read the declaration of.

  Returns:
      tuple: The parameter descriptions keyed by name, and the required
        parameter names.
  """
  declaration = tool._get_declaration()  # pylint: disable=protected-access
  if declaration.parameters is not None:
    properties = declaration.parameters.properties or {}
    descriptions = {
        name: schema.description for name, schema in properties.items()
    }
    required = declaration.parameters.required
  else:
    schema = declaration.parameters_json_schema or {}
    properties = schema.get("properties") or {}
    descriptions = {
        name: prop.get("description") for name, prop in properties.items()
    }
    required = schema.get("required")
  return descriptions, list(required or [])


def get_declaration(settings: BigQueryToolConfig):
  return get_params(BigQueryTool(func=sample_func, tool_settings=settings))


async def run_tool(settings: BigQueryToolConfig, args: dict[str, Any]):
  tool = BigQueryTool(func=sample_func, tool_settings=settings)
  return await tool.run_async(
      args=args, tool_context=mock.Mock(spec=ToolContext)
  )


@pytest.mark.usefixtures("declaration_shape")
def test_declaration_unchanged_without_defaults():
  """Ids stay mandatory when the config does not supply defaults."""
  descriptions, required = get_declaration(BigQueryToolConfig())

  assert set(required) == {"project_id", "dataset_id", "table_id"}
  assert "Defaults to" not in (descriptions["project_id"] or "")


@pytest.mark.usefixtures("declaration_shape")
def test_declaration_drops_defaulted_ids_from_required():
  """A defaulted id is optional for the model and documents its default."""
  descriptions, required = get_declaration(
      BigQueryToolConfig(
          default_project_id="my-project", default_dataset_id="my-dataset"
      )
  )

  assert required == ["table_id"]
  assert (
      "Defaults to 'my-project' when not provided."
      in descriptions["project_id"]
  )
  assert (
      "Defaults to 'my-dataset' when not provided."
      in descriptions["dataset_id"]
  )


@pytest.mark.usefixtures("declaration_shape")
def test_declaration_drops_only_the_configured_id():
  """Only the ids that have a configured default become optional."""
  _, required = get_declaration(
      BigQueryToolConfig(default_project_id="my-project")
  )

  assert set(required) == {"dataset_id", "table_id"}


@pytest.mark.usefixtures("declaration_shape")
def test_declaration_of_a_defaulted_tool_does_not_leak():
  """One tool's defaults must not reach another built on the same function.

  The declarations are built through a cache keyed on the function, so a tool
  that edits its own declaration in place could corrupt every later tool.
  """
  get_declaration(
      BigQueryToolConfig(
          default_project_id="my-project", default_dataset_id="my-dataset"
      )
  )

  descriptions, required = get_declaration(BigQueryToolConfig())

  assert set(required) == {"project_id", "dataset_id", "table_id"}
  assert "Defaults to" not in (descriptions["project_id"] or "")


def test_tool_without_a_signature():
  """A callable the toolset cannot introspect is left alone, not rejected."""
  tool = BigQueryTool(
      func=None,
      tool_settings=BigQueryToolConfig(default_project_id="my-project"),
  )

  assert not tool._arg_defaults  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_run_fills_in_defaults():
  """Omitted ids are filled in from the config before the tool runs."""
  result = await run_tool(
      BigQueryToolConfig(
          default_project_id="my-project", default_dataset_id="my-dataset"
      ),
      {"table_id": "my-table"},
  )

  assert result == {
      "project_id": "my-project",
      "dataset_id": "my-dataset",
      "table_id": "my-table",
  }


@pytest.mark.asyncio
async def test_run_prefers_the_model_supplied_ids():
  """An id the model does supply wins over the configured default."""
  result = await run_tool(
      BigQueryToolConfig(
          default_project_id="my-project", default_dataset_id="my-dataset"
      ),
      {
          "project_id": "other-project",
          "dataset_id": "other-dataset",
          "table_id": "my-table",
      },
  )

  assert result == {
      "project_id": "other-project",
      "dataset_id": "other-dataset",
      "table_id": "my-table",
  }


@pytest.mark.usefixtures("declaration_shape")
@pytest.mark.asyncio
async def test_toolset_applies_defaults():
  """The toolset wires the defaults through to the tools it hands out."""
  toolset = BigQueryToolset(
      credentials_config=BigQueryCredentialsConfig(
          client_id="abc", client_secret="def"
      ),
      tool_filter=["get_table_info", "list_dataset_ids"],
      bigquery_tool_config=BigQueryToolConfig(
          default_project_id="my-project", default_dataset_id="my-dataset"
      ),
  )

  tools = {tool.name: tool for tool in await toolset.get_tools()}

  # get_table_info takes both ids, so only the table id is left mandatory.
  _, required = get_params(tools["get_table_info"])
  assert required == ["table_id"]

  # list_dataset_ids only takes a project id, so nothing is left mandatory.
  _, required = get_params(tools["list_dataset_ids"])
  assert not required
