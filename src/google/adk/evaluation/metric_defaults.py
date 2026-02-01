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

from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo


def get_default_metric_info(
    metric_name: str, description: str = ""
) -> MetricInfo:
  """Returns a default MetricInfo for a metric."""
  return MetricInfo(
      metric_name=metric_name,
      description=description,
      metric_value_info=MetricValueInfo(
          interval=Interval(min_value=0.0, max_value=1.0)
      ),
  )
