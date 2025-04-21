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

from google.adk.telemetry.setup import genai_sdk_instrumentation
from google.adk.telemetry.setup import requests_instrumentation
from google.adk.telemetry.setup import logs_wiring
from google.adk.telemetry.setup import metrics_wiring
from google.adk.telemetry.setup import trace_wiring


def _setup_instrumentation():
    genai_sdk_instrumentation.setup_google_genai_instrumentation()
    requests_instrumentation.setup_requests_instrumentation()


def _setup_wiring(
    extra_trace_processors = None,
    extra_logs_processors = None,
    extra_metric_readers = None):
    logs_wiring.setup_logs_wiring(additional_processors=extra_logs_processors)
    metrics_wiring.setup_metrics_wiring(additional_readers=extra_metric_readers)
    trace_wiring.setup_traces_wiring(additional_processors=extra_trace_processors)


def setup_telemetry(
    extra_trace_processors = None,
    extra_logs_processors = None,
    extra_metric_readers = None):
    _setup_instrumentation()
    _setup_wiring(
        extra_trace_processors=extra_trace_processors,
        extra_logs_processors=extra_logs_processors,
        extra_metric_readers=extra_metric_readers)
