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

"""Side-by-side demo comparing BigQuery Agent Analytics vs Cloud Logging.

This demo shows how to log ADK agent events using two different approaches:
1. BigQuery Agent Analytics Plugin - Full analytics with SQL queries
2. Cloud Logging via OpenTelemetry - Distributed tracing integration

See README.md for detailed comparison and usage instructions.
"""

from . import agent
