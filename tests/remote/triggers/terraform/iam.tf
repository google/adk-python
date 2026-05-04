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
# IAM bindings and Service Accounts for Cloud Run invokers.
# ---------------------------------------------------------------------------

# Service account for the Cloud Run agent itself.
resource "google_service_account" "cloud_run" {
  account_id   = "${local.name_prefix}-run-${local.suffix}"
  display_name = "ADK Trigger Test - Cloud Run Agent"
  project      = var.project_id
}

# Service account for Pub/Sub to invoke Cloud Run.
resource "google_service_account" "pubsub_invoker" {
  account_id   = "${local.name_prefix}-ps-${local.suffix}"
  display_name = "ADK Trigger Test - Pub/Sub Invoker"
  project      = var.project_id
}

# Service account for Eventarc to invoke Cloud Run.
resource "google_service_account" "eventarc_invoker" {
  account_id   = "${local.name_prefix}-ea-${local.suffix}"
  display_name = "ADK Trigger Test - Eventarc Invoker"
  project      = var.project_id
}

resource "google_cloud_run_v2_service_iam_member" "pubsub_invoker" {
  name     = data.google_cloud_run_v2_service.trigger_agent.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_invoker.email}"
}

resource "google_cloud_run_v2_service_iam_member" "eventarc_invoker" {
  name     = data.google_cloud_run_v2_service.trigger_agent.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.eventarc_invoker.email}"
}


# Eventarc requires the receiver role on the invoker service account.
resource "google_project_iam_member" "eventarc_event_receiver" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.eventarc_invoker.email}"
}

data "google_project" "project" {
  project_id = var.project_id
}

# Grant the Pub/Sub service agent permission to create OIDC tokens for the pubsub_invoker SA
resource "google_service_account_iam_member" "pubsub_token_creator" {
  service_account_id = google_service_account.pubsub_invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Grant the Pub/Sub service agent permission to create OIDC tokens for the eventarc_invoker SA
resource "google_service_account_iam_member" "eventarc_token_creator" {
  service_account_id = google_service_account.eventarc_invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}
