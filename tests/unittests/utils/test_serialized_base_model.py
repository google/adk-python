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

"""Tests for SerializedBaseModel."""

from google.adk.utils._serialized_base_model import SerializedBaseModel


class MyModel(SerializedBaseModel):
  test_field: str


def test_model_dump_json_by_alias_default():
  model = MyModel(test_field="value")
  json_str = model.model_dump_json()
  assert "testField" in json_str
  assert "test_field" not in json_str


def test_model_dump_json_by_alias_false():
  model = MyModel(test_field="value")
  json_str = model.model_dump_json(by_alias=False)
  assert "test_field" in json_str
  assert "testField" not in json_str
