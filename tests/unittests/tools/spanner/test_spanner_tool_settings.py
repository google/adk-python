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
from google.adk.tools.spanner.settings import Capabilities
from google.adk.tools.spanner.settings import QueryResultMode
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.settings import SpannerVectorStoreSettings
from pydantic import ValidationError
import pytest


@pytest.fixture(autouse=True)
def reset_warned_features():
  """Reset warned features before each test."""
  _WARNED_FEATURES.clear()


def common_spanner_vector_store_settings(vector_length=None):
  return {
      "project_id": "test-project",
      "instance_id": "test-instance",
      "database_id": "test-database",
      "table_name": "test-table",
      "content_column": "test-content-column",
      "embedding_column": "test-embedding-column",
      "vector_length": 128 if vector_length is None else vector_length,
  }


def test_spanner_tool_settings_experimental_warning():
  """Test SpannerToolSettings experimental warning."""
  with warnings.catch_warnings(record=True) as w:
    SpannerToolSettings()
    assert len(w) == 1
    assert "SPANNER_TOOL_SETTINGS is enabled." in str(w[0].message)


def test_spanner_vector_store_settings_all_fields_present():
  """Test SpannerVectorStoreSettings with all required fields present."""
  settings = SpannerVectorStoreSettings(
      **common_spanner_vector_store_settings(),
      vertex_ai_embedding_model_name="test-embedding-model",
  )
  assert settings is not None
  assert settings.selected_columns == ["test-content-column"]
  assert settings.vertex_ai_embedding_model_name == "test-embedding-model"


def test_spanner_vector_store_settings_missing_embedding_model_name():
  """Test SpannerVectorStoreSettings with missing vertex_ai_embedding_model_name."""
  with pytest.raises(ValidationError) as excinfo:
    SpannerVectorStoreSettings(**common_spanner_vector_store_settings())
  assert "Field required" in str(excinfo.value)
  assert "vertex_ai_embedding_model_name" in str(excinfo.value)


def test_spanner_vector_store_settings_invalid_vector_length():
  """Test SpannerVectorStoreSettings with invalid vector_length."""
  with pytest.raises(ValidationError) as excinfo:
    SpannerVectorStoreSettings(
        **common_spanner_vector_store_settings(vector_length=0),
        vertex_ai_embedding_model_name="test-embedding-model",
    )
  assert "Invalid vector length in the Spanner vector store settings." in str(
      excinfo.value
  )


@pytest.mark.parametrize(
    "settings_args, expected_rows, expected_mode",
    [
        ({}, 50, QueryResultMode.DEFAULT),
        (
            {
                "capabilities": [Capabilities.DATA_READ],
                "max_executed_query_result_rows": 100,
                "query_result_mode": QueryResultMode.DICT_LIST,
            },
            100,
            QueryResultMode.DICT_LIST,
        ),
    ],
)
def test_spanner_tool_settings(settings_args, expected_rows, expected_mode):
  """Test SpannerToolSettings with different values."""
  settings = SpannerToolSettings(**settings_args)
  assert settings.capabilities == [Capabilities.DATA_READ]
  assert settings.max_executed_query_result_rows == expected_rows
  assert settings.query_result_mode == expected_mode
