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

import warnings

from google.adk.features._feature_registry import _WARNED_FEATURES
from google.adk.tools.bigquery.config import BigQueryToolConfig
import pytest


@pytest.fixture(autouse=True)
def reset_warned_features():
  """Reset warned features before each test."""
  _WARNED_FEATURES.clear()


def test_bigquery_tool_config_experimental_warning():
  """Test BigQueryToolConfig experimental warning."""
  with warnings.catch_warnings(record=True) as w:
    BigQueryToolConfig()
    assert len(w) == 1
    assert "BIG_QUERY_TOOL_CONFIG is enabled." in str(w[0].message)


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
  config = BigQueryToolConfig()
  assert config.max_query_result_rows == 50


def test_bigquery_tool_config_max_query_result_rows_custom():
  """Test BigQueryToolConfig max_query_result_rows custom value."""
  config = BigQueryToolConfig(max_query_result_rows=100)
  assert config.max_query_result_rows == 100


def test_bigquery_tool_config_valid_maximum_bytes_billed():
  """Test BigQueryToolConfig raises exception with valid max bytes billed."""
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


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(
            {"environment": "test", "team": "data"},
            id="valid-labels",
        ),
        pytest.param(
            {},
            id="empty-labels",
        ),
        pytest.param(
            None,
            id="none-labels",
        ),
    ],
)
def test_bigquery_tool_config_valid_labels(labels):
  """Test BigQueryToolConfig accepts valid labels."""
  config = BigQueryToolConfig(job_labels=labels)
  assert config.job_labels == labels


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        pytest.param(
            "invalid",
            "Input should be a valid dictionary",
            id="invalid-type",
        ),
        pytest.param(
            {123: "value"},
            "Input should be a valid string",
            id="non-str-key",
        ),
        pytest.param(
            {"key": 123},
            "Input should be a valid string",
            id="non-str-value",
        ),
        pytest.param(
            {"": "value"},
            "Label keys cannot be empty",
            id="empty-label-key",
        ),
    ],
)
def test_bigquery_tool_config_invalid_labels(labels, message):
  """Test BigQueryToolConfig raises an exception with invalid labels."""
  with pytest.raises(
      ValueError,
      match=message,
  ):
    BigQueryToolConfig(job_labels=labels)
