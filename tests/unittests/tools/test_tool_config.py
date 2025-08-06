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

from google.adk.tools import VertexAiSearchTool
from google.adk.tools.tool_configs import ToolConfig
from google.genai import types
import yaml


def test_vertex_ai_search_tool_config():
  yaml_content = """\
name: VertexAiSearchTool
args:
  data_store_specs:
    - data_store: projects/my-project/locations/us-central1/collections/my-collection/dataStores/my-datastore1
      filter: filter1
    - data_store: projects/my-project/locations/us-central1/collections/my-collection/dataStores/my-dataStore2
      filter: filter2
  filter: filter
  max_results: 10
  search_engine_id: projects/my-project/locations/us-central1/collections/my-collection/engines/my-engine
  """
  config_data = yaml.safe_load(yaml_content)
  config = ToolConfig.model_validate(config_data)

  tool = VertexAiSearchTool.from_config(config.args, "")
  assert isinstance(tool, VertexAiSearchTool)
  assert isinstance(tool.data_store_specs[0], types.VertexAISearchDataStoreSpec)
  assert (
      tool.data_store_specs[0].data_store
      == "projects/my-project/locations/us-central1/collections/my-collection/dataStores/my-datastore1"
  )
  assert tool.data_store_specs[0].filter == "filter1"
  assert isinstance(tool.data_store_specs[0], types.VertexAISearchDataStoreSpec)
  assert (
      tool.data_store_specs[1].data_store
      == "projects/my-project/locations/us-central1/collections/my-collection/dataStores/my-dataStore2"
  )
  assert tool.data_store_specs[1].filter == "filter2"
  assert tool.filter == "filter"
  assert tool.max_results == 10
  assert (
      tool.search_engine_id
      == "projects/my-project/locations/us-central1/collections/my-collection/engines/my-engine"
  )
