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

from google.adk.evaluation.metric_defaults import get_default_metric_info


def test_get_default_metric_info():
  metric_info = get_default_metric_info("my_metric", "test description")

  assert metric_info.metric_name == "my_metric"
  assert metric_info.description == "test description"
  assert metric_info.metric_value_info.interval.min_value == 0.0
  assert metric_info.metric_value_info.interval.max_value == 1.0
