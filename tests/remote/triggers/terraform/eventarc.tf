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
# Eventarc: separate Pub/Sub topic as event source → Cloud Run /trigger/eventarc
# ---------------------------------------------------------------------------

# A dedicated topic that acts as the Eventarc event source.
# Publishing to this topic triggers the Eventarc → Cloud Run pipeline.
resource "google_pubsub_topic" "eventarc_source" {
  name    = "${local.name_prefix}-eventarc-${local.suffix}"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_eventarc_trigger" "trigger_test" {
  name     = "${local.name_prefix}-${local.suffix}"
  location = var.region
  project  = var.project_id

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.pubsub.topic.v1.messagePublished"
  }

  transport {
    pubsub {
      topic = google_pubsub_topic.eventarc_source.id
    }
  }

  destination {
    cloud_run_service {
      service = data.google_cloud_run_v2_service.trigger_agent.name
      path    = "/apps/trigger_echo_agent/trigger/eventarc"
      region  = var.region
    }
  }

  service_account = google_service_account.eventarc_invoker.email

  depends_on = [
    google_project_iam_member.eventarc_event_receiver,
    google_project_service.apis,
  ]
}

# ---------------------------------------------------------------------------
# GCS Trigger Test
# ---------------------------------------------------------------------------

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "trigger_test_bucket" {
  name          = "${local.name_prefix}-bucket-${local.suffix}-${random_id.bucket_suffix.hex}"
  location      = var.region
  project       = var.project_id
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.apis]
}

data "google_storage_project_service_account" "gcs_account" {
  project = var.project_id
}

resource "google_project_iam_member" "gcs_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${data.google_storage_project_service_account.gcs_account.email_address}"

  depends_on = [data.google_storage_project_service_account.gcs_account]
}

resource "google_eventarc_trigger" "gcs_trigger" {
  name     = "${local.name_prefix}-gcs-${local.suffix}"
  location = var.region
  project  = var.project_id

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }
  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.trigger_test_bucket.name
  }

  destination {
    cloud_run_service {
      service = data.google_cloud_run_v2_service.trigger_agent.name
      path    = "/apps/trigger_echo_agent/trigger/eventarc"
      region  = var.region
    }
  }

  service_account = google_service_account.eventarc_invoker.email

  depends_on = [
    google_project_iam_member.eventarc_event_receiver,
    google_project_iam_member.gcs_pubsub_publisher,
    google_project_service.apis,
  ]
}
