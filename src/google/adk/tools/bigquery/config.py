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

from pydantic import BaseModel


class BigQueryToolConfig(BaseModel):
  """Configuration for BigQuery tools."""

  # Configure the model to allow arbitrary types like Credentials
  model_config = {"arbitrary_types_allowed": True}

  write_protected: bool = False
  """Protection against unsafe writes.

  One should enable this flag if they don't want the tool to perform any
  operations to create, update or delete resources.
  """
