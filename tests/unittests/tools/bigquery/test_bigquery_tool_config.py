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

from google.adk.tools.bigquery.config import BigQueryToolConfig
import pytest


def test_bigquery_tool_config_experimental_warning():
  """Test BigQueryToolConfig experimental warning."""
  with pytest.warns(
      UserWarning,
      match="Config defaults may have breaking change in the future.",
  ):
    BigQueryToolConfig()


def test_bigquery_tool_config_invalid_property():
  """Test BigQueryToolConfig raises exception when setting invalid property."""
  with pytest.raises(
      ValueError,
  ):
    BigQueryToolConfig(non_existent_field="some value")


def test_bigquery_tool_config_invalid_application_name():
  """Test BigQueryToolConfig raises exception with invalid application name."""
  with pytest.raises(
      ValueError,
      match="Application name should not contain spaces.",
  ):
    BigQueryToolConfig(application_name="my agent")


def test_bigquery_tool_config_max_query_result_rows_default():
  """Test BigQueryToolConfig max_query_result_rows default value."""
  with pytest.warns(UserWarning):
    config = BigQueryToolConfig()
  assert config.max_query_result_rows == 50


def test_bigquery_tool_config_max_query_result_rows_custom():
  """Test BigQueryToolConfig max_query_result_rows custom value."""
  with pytest.warns(UserWarning):
    config = BigQueryToolConfig(max_query_result_rows=100)
  assert config.max_query_result_rows == 100


def test_bigquery_tool_config_valid_maximum_bytes_billed():
  """Test BigQueryToolConfig raises exception with valid max bytes billed."""
  with pytest.warns(UserWarning):
    config = BigQueryToolConfig(maximum_bytes_billed=10_485_760)
  assert config.maximum_bytes_billed == 10_485_760


def test_bigquery_tool_config_invalid_maximum_bytes_billed():
  """Test BigQueryToolConfig raises exception with invalid max bytes billed."""
  with pytest.raises(
      ValueError,
      match=(
          "In BigQuery on-demand pricing, charges are rounded up to the nearest"
          " MB, with a minimum 10 MB data processed per table referenced by the"
          " query, and with a minimum 10 MB data processed per query. So"
          " max_bytes_billed must be set >=10485760."
      ),
  ):
    BigQueryToolConfig(maximum_bytes_billed=10_485_759)
