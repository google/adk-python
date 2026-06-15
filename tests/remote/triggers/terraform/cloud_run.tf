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

# ---------------------------------------------------------------------------
# ADK Agent for trigger testing.
#
# This configuration references a Cloud Run service that has been deployed
# manually (e.g. via `gcloud run deploy`) before running Terraform.
# ---------------------------------------------------------------------------

locals {
  service_name = var.service_name != null ? var.service_name : "${local.name_prefix}-${local.suffix}"
}

# Read the service back as a data source so other resources can reference
# its URL and attributes.
data "google_cloud_run_v2_service" "trigger_agent" {
  name     = local.service_name
  location = var.region
  project  = var.project_id
}

# No longer using resource "google_cloud_run_v2_service" to avoid
# deployment conflicts with local tools.
